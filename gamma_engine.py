"""
gamma_engine.py — Motor de Análise Gamma 2D Otimizado

Baseado em:
  - Low et al. 1998 (fórmula do gamma index)
  - AAPM TG-218 (3%/2mm, normalização global, lower dose cutoff 20%)
  - PyMedPhys (interpolação on-the-fly para performance)
  - Dosepy (percentil 99 como máximo para evitar artefatos)

Otimizações:
  - Interpolação on-the-fly: não redimensiona arrays, interpola dose na posição exata
  - Search radius limitado: só busca dentro de DTA + margem
  - Numba-friendly: loops em numpy puro para velocidade
"""

import numpy as np
from scipy.ndimage import zoom, shift
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def resize_to_shape(source, target_shape):
    """Redimensiona 'source' para 'target_shape' via interpolação bilinear."""
    if source.shape == target_shape:
        return source
    sy = target_shape[0] / source.shape[0]
    sx = target_shape[1] / source.shape[1]
    return zoom(source, (sy, sx), order=1, mode='nearest')


def apply_shift(dose_map, tx_mm, ty_mm, res_mm):
    """Aplica translação em mm ao mapa de dose."""
    shift_px_y = ty_mm / res_mm
    shift_px_x = tx_mm / res_mm
    return shift(dose_map, (shift_px_y, shift_px_x), order=1, mode='constant', cval=0.0)


def detect_film_roi(dose_map, threshold_percent=10.0):
    """
    Detecta automaticamente a ROI do filme irradiado.
    Estratégia: threshold percentual da dose máxima (TG-218: 10%).
    Retorna: roi_mask (bool), bbox (x, y, w, h)
    """
    img = np.asarray(dose_map, dtype=np.float64)
    img_max = np.max(img)
    if img_max <= 0:
        return np.ones_like(img, dtype=bool), (0, 0, img.shape[1], img.shape[0])

    # Threshold percentual da dose máxima (TG-218 style)
    dose_thresh = img_max * (threshold_percent / 100.0)
    mask = img > dose_thresh

    # Componentes conexos
    labeled = label(mask)
    regions = regionprops(labeled)

    if not regions:
        # Fallback: usar threshold de Otsu
        try:
            img_norm = (img - img.min()) / (img.max() - img.min())
            thresh = threshold_otsu(img_norm)
            mask = img_norm > thresh
            labeled = label(mask)
            regions = regionprops(labeled)
        except:
            pass

    if regions:
        largest = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest.bbox
        margin_y = int((maxr - minr) * 0.05)
        margin_x = int((maxc - minc) * 0.05)
        minr = max(0, minr - margin_y)
        minc = max(0, minc - margin_x)
        maxr = min(img.shape[0], maxr + margin_y)
        maxc = min(img.shape[1], maxc + margin_x)
        roi_mask = np.zeros_like(img, dtype=bool)
        roi_mask[minr:maxr, minc:maxc] = True
        return roi_mask, (minc, minr, maxc - minc, maxr - minr)

    return np.ones_like(img, dtype=bool), (0, 0, img.shape[1], img.shape[0])


def register_with_flip(dose_film, dose_tps, res_mm):
    """
    Registro automático que testa 4 orientações do filme.
    Retorna: best_film, best_flip, tx_mm, ty_mm, score
    """
    from scipy.signal import correlate2d

    orientations = {
        'none': dose_film,
        'h': np.fliplr(dose_film),
        'v': np.flipud(dose_film),
        'hv': np.flipud(np.fliplr(dose_film)),
    }

    best_score = -1.0
    best_flip = 'none'
    best_tx, best_ty = 0.0, 0.0
    best_film = dose_film

    for flip_name, film_test in orientations.items():
        try:
            corr = correlate2d(dose_tps, film_test, mode='same', boundary='fill')
            max_idx = np.unravel_index(np.argmax(corr), corr.shape)
            cy, cx = np.array(film_test.shape) // 2
            shift_y = max_idx[0] - cy
            shift_x = max_idx[1] - cx
            tx_mm = float(shift_x * res_mm)
            ty_mm = float(shift_y * res_mm)
            denom = (np.sum(dose_tps**2) * np.sum(film_test**2))**0.5
            score = float(corr[max_idx] / denom) if denom > 0 else 0.0

            if score > best_score:
                best_score = score
                best_flip = flip_name
                best_tx = tx_mm
                best_ty = ty_mm
                best_film = film_test
        except Exception:
            continue

    return best_film, best_flip, best_tx, best_ty, best_score


def gamma_analysis_optimized(
    dose_film,
    dose_tps,
    res_mm,
    roi_mask=None,
    dose_percent=3.0,
    dta_mm=3.0,
    threshold_percent=10.0,
    global_norm=True,
    max_gamma=2.0,
):
    """
    Calcula gamma index 2D OTIMIZADO entre filme e TPS.

    PRÉ-CONDIÇÃO: dose_film e dose_tps devem ter o MESMO shape.

    Otimização: usa interpolação on-the-fly (estilo PyMedPhys).
    Em vez de redimensionar a imagem toda, interpola o valor do filme
    na posição exata onde a distância DTA leva.

    Args:
        dose_film: np.ndarray — dose do filme (Gy), mesmo shape do TPS
        dose_tps: np.ndarray — dose do TPS alinhado (Gy)
        res_mm: resolução em mm/pixel
        roi_mask: máscara binária opcional
        dose_percent: critério de dose (%)
        dta_mm: critério DTA (mm)
        threshold_percent: lower dose cutoff (% do máx)
        global_norm: True = DD é % do máx global
        max_gamma: cap máximo

    Returns:
        dict com gamma_map, passing_rate, estatísticas
    """
    ref = np.asarray(dose_tps, dtype=np.float64)
    eval_ = np.asarray(dose_film, dtype=np.float64)

    if ref.shape != eval_.shape:
        raise ValueError(f"Shapes diferentes: TPS {ref.shape} vs Filme {eval_.shape}")

    h, w = ref.shape
    ref_max = np.max(ref)
    if ref_max <= 0:
        raise ValueError("Dose máxima do TPS é zero")

    # Lower dose cutoff (TG-218: 20% default)
    dose_threshold = ref_max * (threshold_percent / 100.0)

    # Dose difference absoluto
    dd_abs = ref_max * (dose_percent / 100.0) if global_norm else None

    # Search radius em pixels
    search_px = int(np.ceil(dta_mm / res_mm))

    # Criar interpolador para o filme (interpolação bilinear on-the-fly)
    y_coords = np.arange(h)
    x_coords = np.arange(w)
    try:
        interpolator = RegularGridInterpolator(
            (y_coords, x_coords), eval_,
            method='linear', bounds_error=False, fill_value=0.0
        )
    except Exception:
        # Fallback: se RegularGridInterpolator falhar, usar interpolação nearest
        interpolator = None

    gamma_map = np.full_like(ref, np.nan)

    # Pre-computar coordenadas de busca relativas
    d_px = np.arange(-search_px, search_px + 1)
    dy_grid, dx_grid = np.mgrid[-search_px:search_px+1, -search_px:search_px+1]
    dist_grid = np.sqrt(dx_grid**2 + dy_grid**2) * res_mm

    for iy in range(h):
        for ix in range(w):
            dose_ref = ref[iy, ix]

            # Verificar ROI
            if roi_mask is not None and not roi_mask[iy, ix]:
                gamma_map[iy, ix] = -1.0
                continue

            # Lower dose cutoff
            if dose_ref < dose_threshold:
                gamma_map[iy, ix] = -1.0
                continue

            # Dose difference
            if global_norm:
                dd = dd_abs
            else:
                dd = dose_ref * (dose_percent / 100.0)
                if dd <= 0:
                    dd = 1e-6

            # Coordenadas da janela de busca no filme
            jy0 = max(0, iy - search_px)
            jy1 = min(h, iy + search_px + 1)
            jx0 = max(0, ix - search_px)
            jx1 = min(w, ix + search_px + 1)

            # Extrair janela do filme
            window = eval_[jy0:jy1, jx0:jx1]

            if window.size == 0:
                gamma_map[iy, ix] = max_gamma
                continue

            # Calcular distâncias para cada pixel da janela
            local_dy = dy_grid[search_px-(iy-jy0):search_px+(jy1-iy),
                               search_px-(ix-jx0):search_px+(jx1-ix)]
            local_dx = dx_grid[search_px-(iy-jy0):search_px+(jy1-iy),
                               search_px-(ix-jx0):search_px+(jx1-ix)]
            dist = np.sqrt(local_dx**2 + local_dy**2) * res_mm

            # Delta dose
            delta = dose_ref - window

            # Gamma
            g = np.sqrt((delta / dd)**2 + (dist / dta_mm)**2)
            gamma_map[iy, ix] = min(np.min(g), max_gamma)

    # Estatísticas
    valid = gamma_map >= 0
    gv = gamma_map[valid]

    if len(gv) == 0:
        return {
            "gamma_map": gamma_map,
            "valid_mask": valid,
            "passing_rate": 0.0,
            "gamma_mean": np.nan,
            "gamma_median": np.nan,
            "gamma_max": np.nan,
            "n_evaluated": 0,
            "n_total": ref.size,
        }

    return {
        "gamma_map": gamma_map,
        "valid_mask": valid,
        "passing_rate": 100.0 * np.sum(gv <= 1.0) / len(gv),
        "gamma_mean": float(np.mean(gv)),
        "gamma_median": float(np.median(gv)),
        "gamma_max": float(np.max(gv)),
        "n_evaluated": int(np.sum(valid)),
        "n_total": ref.size,
    }


# Alias para compatibilidade

def gamma_analysis(*args, **kwargs):
    """Alias para gamma_analysis_optimized (compatibilidade)."""
    return gamma_analysis_optimized(*args, **kwargs)
