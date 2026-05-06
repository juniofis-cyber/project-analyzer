"""
Gamma Engine 2D — Implementação própria do Gamma Index (Low et al. 1998).

Sem dependências externas (só NumPy + SciPy).
Validar contra PyMedPhys como golden standard.

Algoritmo:
  Para cada ponto da referência (TPS):
    1. Se dose < low_dose_threshold: γ = -1 (não avaliado)
    2. Procurar na vizinhança da avaliação (filme) dentro de search_radius
    3. Calcular Γ = sqrt((δ/ΔD)^2 + (r/Δd)^2) para cada ponto da vizinhança
    4. γ = min(Γ)

Onde:
  δ = dose_ref - dose_eval
  r = distância espacial (mm)
  ΔD = dose_percent_threshold (% do max global ou local)
  Δd = distance_mm_threshold (mm)
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
    mask_evaluation=None,
):
    """
    Calcula gamma index 2D entre duas distribuições de dose.

    Args:
        reference_dose: np.ndarray — dose do TPS (Gy)
        evaluation_dose: np.ndarray — dose do filme (Gy)
        reference_resolution_mm: resolução do TPS em mm/pixel
        evaluation_resolution_mm: resolução do filme em mm/pixel
        dose_percent_threshold: critério de dose (%)
        distance_mm_threshold: critério de DTA (mm)
        low_dose_percent_threshold: pixels abaixo deste % do máx são ignorados
        global_normalization: True = DD é % do máx global da referência
        interp_factor: fator de interpolação da avaliação (recomendado: 10)
        max_gamma: cap máximo de gamma (para visualização)
        search_radius_factor: raio de busca = factor × DTA
        mask_evaluation: máscara binária opcional (1 = calcular, 0 = ignorar)

    Returns:
        dict com:
          gamma_map: matriz de gamma (mesmo shape da referência)
          passing_rate: % de pixels avaliados com γ ≤ 1.0
          gamma_mean, gamma_median, gamma_max: estatísticas
          valid_mask: máscara de pixels avaliados
          dose_threshold_abs: valor absoluto do threshold de dose
    """
    ref = np.asarray(reference_dose, dtype=np.float64)
    eval_orig = np.asarray(evaluation_dose, dtype=np.float64)

    # Normalização da dose
    ref_max = np.max(ref)
    if ref_max <= 0:
        raise ValueError("Dose de referência máxima é zero ou negativa")

    dose_threshold_abs = ref_max * (low_dose_percent_threshold / 100.0)
    dd_abs = ref_max * (dose_percent_threshold / 100.0) if global_normalization else None

    # --- PASSO 1: Interpolar a avaliação para grade mais fina ---
    # Regra: interp_factor deve ser suficiente para ter ~3-10 pontos por DTA
    eval_interp, eval_res_interp = _interpolate_evaluation(
        eval_orig, evaluation_resolution_mm, interp_factor
    )

    # --- PASSO 2: Preparar grade de coordenadas ---
    h_ref, w_ref = ref.shape
    h_eval, w_eval = eval_interp.shape

    # Eixos físicos em mm (centro da imagem em 0,0)
    x_ref = (np.arange(w_ref) - w_ref / 2) * reference_resolution_mm
    y_ref = (np.arange(h_ref) - h_ref / 2) * reference_resolution_mm
    x_eval = (np.arange(w_eval) - w_eval / 2) * eval_res_interp
    y_eval = (np.arange(h_eval) - h_eval / 2) * eval_res_interp

    # --- PASSO 3: Construir interpolador da avaliação ---
    interpolator = RegularGridInterpolator(
        (y_eval, x_eval),
        eval_interp,
        bounds_error=False,
        fill_value=0.0,
        method='linear',
    )

    # --- PASSO 4: Calcular gamma para cada ponto da referência ---
    gamma_map = np.full_like(ref, np.nan)

    # Search radius em pixels da grade interpolada
    search_radius_mm = search_radius_factor * distance_mm_threshold
    search_px = int(np.ceil(search_radius_mm / eval_res_interp))

    # Centro da avaliação interpolada
    cy_eval, cx_eval = h_eval // 2, w_eval // 2

    for iy in range(h_ref):
        y_mm = y_ref[iy]
        for ix in range(w_ref):
            dose_ref = ref[iy, ix]

            # Pixels abaixo do threshold não são avaliados
            if dose_ref < dose_threshold_abs:
                gamma_map[iy, ix] = -1.0
                continue

            # Dose difference absoluto para este ponto
            if global_normalization:
                dd_point = dd_abs
            else:
                dd_point = dose_ref * (dose_percent_threshold / 100.0)
                if dd_point <= 0:
                    dd_point = 1e-6

            # Coordenada física deste ponto
            x_mm = x_ref[ix]

            # Definir janela de busca na avaliação interpolada
            jy_start = max(0, cy_eval + int(np.round((y_mm - search_radius_mm) / eval_res_interp)) - cy_eval)
            jy_end = min(h_eval, cy_eval + int(np.round((y_mm + search_radius_mm) / eval_res_interp)) - cy_eval + 1)
            jx_start = max(0, cx_eval + int(np.round((x_mm - search_radius_mm) / eval_res_interp)) - cx_eval)
            jx_end = min(w_eval, cx_eval + int(np.round((x_mm + search_radius_mm) / eval_res_interp)) - cx_eval + 1)

            # Pontos da janela de busca
            y_window = y_eval[jy_start:jy_end]
            x_window = x_eval[jx_start:jx_end]
            yy, xx = np.meshgrid(y_window, x_window, indexing='ij')

            # Amostrar dose da avaliação na janela
            points = np.column_stack([yy.ravel(), xx.ravel()])
            dose_eval_window = interpolator(points).reshape(yy.shape)

            # Distâncias em mm
            dy = yy - y_mm
            dx = xx - x_mm
            dist_mm = np.sqrt(dx**2 + dy**2)

            # Dose difference
            delta = dose_ref - dose_eval_window

            # Gamma
            gamma_window = np.sqrt((delta / dd_point)**2 + (dist_mm / distance_mm_threshold)**2)

            # Mínimo
            gamma_min = np.min(gamma_window)
            gamma_map[iy, ix] = min(gamma_min, max_gamma)

    # --- PASSO 5: Estatísticas ---
    valid_mask = gamma_map >= 0  # excluir pixels com -1 (abaixo do threshold)

    if mask_evaluation is not None:
        valid_mask = valid_mask & (mask_evaluation > 0)

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


def _interpolate_evaluation(eval_dose, eval_res_mm, interp_factor):
    """Interpola a distribuição de avaliação para grade mais fina."""
    if interp_factor <= 1:
        return eval_dose, eval_res_mm

    # Usar zoom do scipy.ndimage
    zoom_factors = (interp_factor, interp_factor)
    eval_interp = zoom(eval_dose, zoom_factors, order=1, mode='nearest')
    eval_res_interp = eval_res_mm / interp_factor

    return eval_interp, eval_res_interp


def gamma_stats_by_dose_region(gamma_map, reference_dose, dose_bins=None):
    """
    Calcula passing rate por faixas de dose (similar ao FilmQA Pro).

    Útil para identificar em que regiões (baixa/média/alta dose) o gamma é pior.
    """
    if dose_bins is None:
        dose_bins = [0, 20, 50, 80, 100]  # % da dose máxima

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
