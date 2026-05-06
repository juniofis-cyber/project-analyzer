"""
Gamma Engine 2D — Simplificado e Correto.

Algoritmo:
  1. Interpola o TPS para ter o MESMO tamanho do filme (zoom)
  2. Para cada ponto da grade comum (tamanho do filme):
     a. Se filme ou TPS < threshold: ignora
     b. Calcula dose difference
     c. Procura na vizinhança do TPS (shifted)
     d. γ = min Γ
"""

import numpy as np
from scipy.ndimage import zoom, shift


def resize_to_shape(source, target_shape):
    """Redimensiona 'source' para 'target_shape' via zoom."""
    sy = target_shape[0] / source.shape[0]
    sx = target_shape[1] / source.shape[1]
    return zoom(source, (sy, sx), order=1, mode='nearest')


def apply_shift(dose_map, tx_mm, ty_mm, res_mm):
    """Aplica translação em mm ao mapa de dose (em pixels)."""
    shift_px_y = ty_mm / res_mm
    shift_px_x = tx_mm / res_mm
    return shift(dose_map, (shift_px_y, shift_px_x), order=1, mode='constant', cval=0.0)


def gamma_analysis(
    dose_film,
    dose_tps_resized,
    res_mm,
    dose_percent=3.0,
    dta_mm=3.0,
    threshold_percent=10.0,
    global_norm=True,
    max_gamma=2.0,
):
    """
    Calcula gamma index.
    Ambos os mapas devem ter o MESMO shape e MESMA resolução.
    """
    ref = np.asarray(dose_tps_resized, dtype=np.float64)
    eval_ = np.asarray(dose_film, dtype=np.float64)
    
    assert ref.shape == eval_.shape, f"Shapes diferentes: {ref.shape} vs {eval_.shape}"
    
    ref_max = np.max(ref)
    if ref_max <= 0:
        raise ValueError("Dose máxima do TPS é zero")
    
    threshold = ref_max * (threshold_percent / 100.0)
    dd_abs = ref_max * (dose_percent / 100.0) if global_norm else None
    
    h, w = ref.shape
    gamma_map = np.full_like(ref, np.nan)
    
    # Search radius em pixels
    search_px = int(np.ceil(dta_mm / res_mm))
    
    for iy in range(h):
        for ix in range(w):
            dose_ref = ref[iy, ix]
            
            # ROI: onde TPS tem dose significativa
            if dose_ref < threshold:
                gamma_map[iy, ix] = -1.0
                continue
            
            # Dose difference
            if global_norm:
                dd = dd_abs
            else:
                dd = dose_ref * (dose_percent / 100.0)
                if dd <= 0:
                    dd = 1e-6
            
            # Janela de busca na avaliação (filme)
            jy0 = max(0, iy - search_px)
            jy1 = min(h, iy + search_px + 1)
            jx0 = max(0, ix - search_px)
            jx1 = min(w, ix + search_px + 1)
            
            window = eval_[jy0:jy1, jx0:jx1]
            
            if window.size == 0:
                gamma_map[iy, ix] = max_gamma
                continue
            
            # Distâncias em mm
            yy, xx = np.mgrid[jy0:jy1, jx0:jx1]
            dy = (yy - iy) * res_mm
            dx = (xx - ix) * res_mm
            dist = np.sqrt(dx**2 + dy**2)
            
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
            "gamma_map": gamma_map, "valid_mask": valid,
            "passing_rate": 0.0, "gamma_mean": np.nan,
            "gamma_median": np.nan, "gamma_max": np.nan,
            "n_evaluated": 0, "n_total": ref.size,
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
