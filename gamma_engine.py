"""
Gamma Engine 2D — Implementação própria do Gamma Index (Low et al. 1998).

Sem dependências externas (só NumPy + SciPy).

Algoritmo:
  Para cada ponto da referência (TPS):
    1. Se dose < low_dose_threshold: γ = -1 (não avaliado)
    2. Converter coordenada do TPS para coordenada física (mm)
    3. Aplicar registro inverso para encontrar posição correspondente no filme
    4. Procurar na vizinhança do filme
    5. Calcular Γ = sqrt((δ/ΔD)^2 + (r/Δd)^2)
    6. γ = min(Γ)
"""

import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator


def gamma_analysis_2d(
    reference_dose,
    evaluation_dose,
    reference_resolution_mm,
    evaluation_resolution_mm,
    dose_percent_threshold=3.0,
    distance_mm_threshold=3.0,
    low_dose_percent_threshold=10.0,
    global_normalization=True,
    interp_factor=10,
    max_gamma=2.0,
    search_radius_factor=2.0,
    registration_tx_mm=0.0,
    registration_ty_mm=0.0,
    mask_evaluation=None,
):
    """
    Calcula gamma index 2D entre duas distribuições de dose.

    DIFERENÇA IMPORTANTE: Esta versão NÃO corta a avaliação.
    A referência (TPS) pode ser menor que a avaliação (filme).
    Para cada ponto do TPS, buscamos no filme inteiro.

    Args:
        reference_dose: np.ndarray — dose do TPS (Gy), shape (h_ref, w_ref)
        evaluation_dose: np.ndarray — dose do filme (Gy), shape (h_eval, w_eval)
        reference_resolution_mm: resolução do TPS em mm/pixel
        evaluation_resolution_mm: resolução do filme em mm/pixel
        dose_percent_threshold: critério de dose (%)
        distance_mm_threshold: critério de DTA (mm)
        low_dose_percent_threshold: pixels abaixo deste % do máx são ignorados
        global_normalization: True = DD é % do máx global da referência
        interp_factor: fator de interpolação da avaliação
        max_gamma: cap máximo de gamma
        search_radius_factor: raio de busca = factor × DTA
        registration_tx_mm: translação X aplicada ao filme (mm)
        registration_ty_mm: translação Y aplicada ao filme (mm)
        mask_evaluation: máscara binária opcional (1 = calcular, 0 = ignorar)

    Returns:
        dict com gamma_map, passing_rate, estatísticas, valid_mask
    """
    ref = np.asarray(reference_dose, dtype=np.float64)
    eval_orig = np.asarray(evaluation_dose, dtype=np.float64)

    ref_max = np.max(ref)
    if ref_max <= 0:
        raise ValueError("Dose de referência máxima é zero ou negativa")

    dose_threshold_abs = ref_max * (low_dose_percent_threshold / 100.0)
    dd_abs = ref_max * (dose_percent_threshold / 100.0) if global_normalization else None

    h_ref, w_ref = ref.shape
    h_eval, w_eval = eval_orig.shape

    # --- PASSO 1: Interpolar a avaliação (filme) para grade mais fina ---
    if interp_factor > 1:
        zoom_factors = (interp_factor, interp_factor)
        eval_interp = zoom(eval_orig, zoom_factors, order=1, mode='nearest')
        eval_res_interp = evaluation_resolution_mm / interp_factor
    else:
        eval_interp = eval_orig
        eval_res_interp = evaluation_resolution_mm

    h_eval_i, w_eval_i = eval_interp.shape

    # --- PASSO 2: Eixos físicos em mm (centro em 0,0) ---
    # Referência (TPS)
    x_ref = (np.arange(w_ref) - w_ref / 2) * reference_resolution_mm
    y_ref = (np.arange(h_ref) - h_ref / 2) * reference_resolution_mm

    # Avaliação (filme) — já com interpolação
    x_eval = (np.arange(w_eval_i) - w_eval_i / 2) * eval_res_interp
    y_eval = (np.arange(h_eval_i) - h_eval_i / 2) * eval_res_interp

    # --- PASSO 3: Construir interpolador do filme ---
    interpolator = RegularGridInterpolator(
        (y_eval, x_eval),
        eval_interp,
        bounds_error=False,
        fill_value=0.0,
        method='linear',
    )

    # --- PASSO 4: Calcular gamma para cada ponto da referência ---
    gamma_map = np.full_like(ref, np.nan)

    search_radius_mm = search_radius_factor * distance_mm_threshold

    for iy in range(h_ref):
        y_mm_tps = y_ref[iy]
        for ix in range(w_ref):
            dose_ref = ref[iy, ix]

            # Pixels abaixo do threshold não são avaliados
            if dose_ref < dose_threshold_abs:
                gamma_map[iy, ix] = -1.0
                continue

            # Dose difference absoluto
            if global_normalization:
                dd_point = dd_abs
            else:
                dd_point = dose_ref * (dose_percent_threshold / 100.0)
                if dd_point <= 0:
                    dd_point = 1e-6

            x_mm_tps = x_ref[ix]

            # Converter coordenada TPS para coordenada no filme (aplicar registro inverso)
            # Se o filme foi transladado por (tx, ty), o ponto correspondente no filme é:
            x_mm_film = x_mm_tps - registration_tx_mm
            y_mm_film = y_mm_tps - registration_ty_mm

            # Definir janela de busca no filme
            jy_start = max(0, int(np.round((y_mm_film - search_radius_mm) / eval_res_interp)) + h_eval_i // 2)
            jy_end = min(h_eval_i, int(np.round((y_mm_film + search_radius_mm) / eval_res_interp)) + h_eval_i // 2 + 1)
            jx_start = max(0, int(np.round((x_mm_film - search_radius_mm) / eval_res_interp)) + w_eval_i // 2)
            jx_end = min(w_eval_i, int(np.round((x_mm_film + search_radius_mm) / eval_res_interp)) + w_eval_i // 2 + 1)

            # Verificar se janela é válida
            if jy_start >= jy_end or jx_start >= jx_end:
                # Fora do filme — usar apenas dose difference (sem DTA)
                dose_eval_center = float(interpolator([[y_mm_film, x_mm_film]])[0])
                delta = dose_ref - dose_eval_center
                gamma_map[iy, ix] = min(abs(delta / dd_point), max_gamma)
                continue

            # Pontos da janela
            y_window = y_eval[jy_start:jy_end]
            x_window = x_eval[jx_start:jx_end]
            yy, xx = np.meshgrid(y_window, x_window, indexing='ij')

            # Amostrar dose
            points = np.column_stack([yy.ravel(), xx.ravel()])
            dose_eval_window = interpolator(points).reshape(yy.shape)

            # Distâncias
            dy = yy - y_mm_film
            dx = xx - x_mm_film
            dist_mm = np.sqrt(dx**2 + dy**2)
            delta = dose_ref - dose_eval_window

            # Gamma
            gamma_window = np.sqrt((delta / dd_point)**2 + (dist_mm / distance_mm_threshold)**2)

            if gamma_window.size > 0:
                gamma_min = np.min(gamma_window)
                gamma_map[iy, ix] = min(gamma_min, max_gamma)
            else:
                gamma_map[iy, ix] = max_gamma

    # --- PASSO 5: Estatísticas ---
    valid_mask = gamma_map >= 0

    if mask_evaluation is not None:
        # Redimensionar máscara para o shape da referência se necessário
        if mask_evaluation.shape != ref.shape:
            from scipy.ndimage import zoom as ndzoom
            zoom_y = ref.shape[0] / mask_evaluation.shape[0]
            zoom_x = ref.shape[1] / mask_evaluation.shape[1]
            mask_resized = ndzoom(mask_evaluation.astype(float), (zoom_y, zoom_x), order=0, mode='nearest')
            mask_resized = mask_resized > 0.5
        else:
            mask_resized = mask_evaluation
        valid_mask = valid_mask & mask_resized

    gamma_valid = gamma_map[valid_mask]

    if len(gamma_valid) == 0:
        passing_rate = 0.0
        gamma_mean = gamma_median = gamma_max = np.nan
    else:
        passing_rate = 100.0 * np.sum(gamma_valid <= 1.0) / len(gamma_valid)
        gamma_mean = float(np.mean(gamma_valid))
        gamma_median = float(np.median(gamma_valid))
        gamma_max = float(np.max(gamma_valid))

    return {
        "gamma_map": gamma_map,
        "passing_rate": passing_rate,
        "gamma_mean": gamma_mean,
        "gamma_median": gamma_median,
        "gamma_max": gamma_max,
        "valid_mask": valid_mask,
        "dose_threshold_abs": dose_threshold_abs,
        "reference_max": ref_max,
        "n_evaluated": int(np.sum(valid_mask)),
        "n_total": ref.size,
    }


def gamma_stats_by_dose_region(gamma_map, reference_dose, dose_bins=None):
    """Calcula passing rate por faixas de dose."""
    if dose_bins is None:
        dose_bins = [0, 20, 50, 80, 100]

    ref_max = np.max(reference_dose)
    stats = []

    for i in range(len(dose_bins) - 1):
        low = ref_max * (dose_bins[i] / 100.0)
        high = ref_max * (dose_bins[i + 1] / 100.0)

        mask = (reference_dose >= low) & (reference_dose < high) & (gamma_map >= 0)
        gamma_region = gamma_map[mask]

        if len(gamma_region) > 0:
            stats.append({
                "dose_range_percent": f"{dose_bins[i]}-{dose_bins[i+1]}%",
                "dose_range_gy": f"{low:.2f}-{high:.2f}",
                "n_pixels": len(gamma_region),
                "passing_rate": 100.0 * np.sum(gamma_region <= 1.0) / len(gamma_region),
                "gamma_mean": float(np.mean(gamma_region)),
                "gamma_median": float(np.median(gamma_region)),
                "gamma_max": float(np.max(gamma_region)),
            })

    return stats
