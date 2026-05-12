"""
Registration — Alinhamento automático entre filme e TPS.

Métodos:
  1. Phase Cross-Correlation (scikit-image) — translação automática
  2. Grid search de rotação — se necessário
  3. Fallback: input manual de translação/rotação

Entrada:
  - dose_film: matriz de dose do filme (Gy)
  - dose_tps: matriz de dose do TPS (Gy)
  - res_film: resolução do filme em mm/pixel
  - res_tps: resolução do TPS em mm/pixel

Saída:
  - Transformação (tx, ty, rotation, scale) para alinhar filme no TPS
"""

import numpy as np
from scipy.ndimage import shift, rotate, zoom


def auto_register_dose_maps(
    dose_film,
    dose_tps,
    res_film_mm,
    res_tps_mm,
    allow_rotation=True,
    rotation_range_deg=(-5.0, 5.0),
    rotation_step_deg=0.5,
):
    """
    Registro automático entre mapa de dose do filme e mapa de dose do TPS.

    Estratégia:
      1. Normalizar ambos para mesma faixa (0-1)
      2. Redimensionar para grade comum (mais grossa = mais rápido)
      3. Phase cross-correlation para encontrar translação ótima
      4. Se allow_rotation: grid search de rotação
      5. Retornar transformação

    Returns:
        dict com tx_mm, ty_mm, rotation_deg, scale, correlation_score
    """
    film = np.asarray(dose_film, dtype=np.float64)
    tps = np.asarray(dose_tps, dtype=np.float64)

    # Normalizar
    film_norm = _normalize(film)
    tps_norm = _normalize(tps)

    # Reamostrar para grade comum
    # Estratégia: reamostrar TPS para resolução do filme (mais fina = melhor precisão)
    # Depois crop/pad ambos para mesmo shape
    target_res = min(res_film_mm, res_tps_mm)  # usar resolução mais fina
    
    # Calcular zoom factors para ambos atingirem a resolução alvo
    zoom_f = res_film_mm / target_res
    zoom_t = res_tps_mm / target_res
    
    film_rs = zoom(film_norm, (zoom_f, zoom_f), order=1, mode='nearest')
    tps_rs = zoom(tps_norm, (zoom_t, zoom_t), order=1, mode='nearest')
    
    # Garantir mesmo shape (usar shape do TPS como referência, que é o reference)
    target_shape = tps_rs.shape
    film_rs = _crop_or_pad(film_rs, target_shape)
    tps_rs = _crop_or_pad(tps_rs, target_shape)

    # --- PASSO 1: Phase Cross-Correlation (translação) ---
    try:
        from skimage.registration import phase_cross_correlation
        shift_px, error, _ = phase_cross_correlation(tps_rs, film_rs, upsample_factor=10)
        tx_mm = float(shift_px[1] * target_res)
        ty_mm = float(shift_px[0] * target_res)
        corr_score = float(1.0 - error)
    except ImportError:
        # Fallback: cross-correlation manual (mais lento)
        tx_mm, ty_mm, corr_score = _cross_correlation_manual(tps_rs, film_rs, target_res)
    except Exception as e:
        # Se phase_cross_correlation falhar (ex: shapes incompatíveis), usar fallback
        tx_mm, ty_mm, corr_score = _cross_correlation_manual(tps_rs, film_rs, target_res)

    best_rot = 0.0

    # --- PASSO 2: Grid search de rotação (se necessário) ---
    if allow_rotation and rotation_range_deg[1] > rotation_range_deg[0]:
        best_score = corr_score
        for rot in np.arange(rotation_range_deg[0], rotation_range_deg[1] + rotation_step_deg, rotation_step_deg):
            film_rot = rotate(film_rs, rot, reshape=False, order=1, mode='nearest')
            try:
                from skimage.registration import phase_cross_correlation
                _, error_rot, _ = phase_cross_correlation(tps_rs, film_rot, upsample_factor=5)
                score = 1.0 - error_rot
            except ImportError:
                _, _, score = _cross_correlation_manual(tps_rs, film_rot, target_res)

            if score > best_score:
                best_score = score
                best_rot = rot
                # Atualizar translação
                try:
                    shift_px, _, _ = phase_cross_correlation(tps_rs, film_rot, upsample_factor=5)
                    tx_mm = float(shift_px[1] * target_res)
                    ty_mm = float(shift_px[0] * target_res)
                except ImportError:
                    pass

        corr_score = best_score

    # Limitar translação a valores razoáveis (±100 mm é mais que suficiente)
    max_shift = 100.0
    tx_mm = max(-max_shift, min(max_shift, tx_mm))
    ty_mm = max(-max_shift, min(max_shift, ty_mm))

    return {
        "tx_mm": tx_mm,
        "ty_mm": ty_mm,
        "rotation_deg": float(best_rot),
        "scale": 1.0,
        "correlation_score": corr_score,
        "target_resolution_mm": target_res,
    }


def apply_transform(dose_film, res_film_mm, tx_mm, ty_mm, rotation_deg=0.0, scale=1.0, target_shape=None):
    """
    Aplica transformação (translação + rotação + escala) no mapa de dose do filme.

    Args:
        target_shape: shape desejado (h, w) para o output alinhado
    """
    film = np.asarray(dose_film, dtype=np.float64)

    # Translação em pixels
    shift_px = (ty_mm / res_film_mm, tx_mm / res_film_mm)
    film_shifted = shift(film, shift_px, order=1, mode='nearest')

    # Rotação
    if abs(rotation_deg) > 0.01:
        film_rotated = rotate(film_shifted, rotation_deg, reshape=False, order=1, mode='nearest')
    else:
        film_rotated = film_shifted

    # Escala (se necessário)
    if abs(scale - 1.0) > 0.001 and target_shape is not None:
        zoom_y = target_shape[0] / film_rotated.shape[0] * scale
        zoom_x = target_shape[1] / film_rotated.shape[1] * scale
        film_scaled = zoom(film_rotated, (zoom_y, zoom_x), order=1, mode='nearest')
        # Crop/pad para target_shape
        film_final = _crop_or_pad(film_scaled, target_shape)
    else:
        film_final = film_rotated
        if target_shape is not None and film_final.shape != target_shape:
            film_final = _crop_or_pad(film_final, target_shape)

    return film_final


def detect_film_roi(image, scanner_direction="vertical"):
    """
    Detecta automaticamente a região do filme na imagem escaneada.

    Retorna máscara binária + bounding box.
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    img = np.asarray(image, dtype=np.float64)

    # Se for RGB, usar canal vermelho (mais sensível ao escurecimento)
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = img[:, :, 0]  # Canal vermelho
    elif img.ndim == 3:
        gray = np.mean(img[:, :, :3], axis=2)
    else:
        gray = img

    # Normalizar
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    # Threshold de Otsu
    try:
        thresh = threshold_otsu(gray)
    except:
        thresh = 0.5

    # O filme irradiado é MAIS ESCURO que o fundo do scanner
    # Mas o filme 0 Gy também é diferente do fundo branco
    mask = gray < thresh  # escuro = filme

    # Remover pequenos artefatos
    labeled = label(mask)
    regions = regionprops(labeled)

    if not regions:
        return np.ones_like(gray, dtype=bool), (0, 0, gray.shape[1], gray.shape[0])

    # Selecionar maior região (o filme)
    largest = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox

    roi_mask = np.zeros_like(gray, dtype=bool)
    roi_mask[minr:maxr, minc:maxc] = True

    return roi_mask, (minc, minr, maxc - minc, maxr - minr)


def _normalize(arr):
    """Normaliza array para 0-1."""
    a = np.asarray(arr, dtype=np.float64)
    mn, mx = a.min(), a.max()
    if mx > mn:
        return (a - mn) / (mx - mn)
    return np.zeros_like(a)


def _resample_to_common(film, tps, res_film, res_tps, target_res):
    """Reamostra ambos para mesma resolução espacial E mesmo shape."""
    zoom_film = res_film / target_res
    zoom_tps = res_tps / target_res

    film_rs = zoom(film, (zoom_film, zoom_film), order=1, mode='nearest')
    tps_rs = zoom(tps, (zoom_tps, zoom_tps), order=1, mode='nearest')

    # phase_cross_correlation exige mesmo shape — usar shape do TPS como referência
    target_shape = tps_rs.shape
    film_rs = _crop_or_pad(film_rs, target_shape)
    # TPS pode já estar no shape certo, mas garantir
    tps_rs = _crop_or_pad(tps_rs, target_shape)

    return film_rs, tps_rs


def _cross_correlation_manual(tps, film, res_mm):
    """Cross-correlation manual (fallback se skimage não disponível)."""
    from scipy.signal import correlate2d

    corr = correlate2d(tps, film, mode='same', boundary='fill')
    max_idx = np.unravel_index(np.argmax(corr), corr.shape)
    cy, cx = np.array(film.shape) // 2
    shift_y = max_idx[0] - cy
    shift_x = max_idx[1] - cx

    tx_mm = float(shift_x * res_mm)
    ty_mm = float(shift_y * res_mm)

    # Score aproximado
    score = float(corr[max_idx] / (np.sum(tps**2) * np.sum(film**2))**0.5)

    return tx_mm, ty_mm, score


def _crop_or_pad(arr, target_shape):
    """Corta ou preenche array para atingir target_shape."""
    h, w = arr.shape
    th, tw = target_shape

    result = np.zeros((th, tw), dtype=arr.dtype)

    # Índices para copiar
    y_start = max(0, (th - h) // 2)
    x_start = max(0, (tw - w) // 2)
    y_src_start = max(0, (h - th) // 2)
    x_src_start = max(0, (w - tw) // 2)

    copy_h = min(h, th)
    copy_w = min(w, tw)

    result[y_start:y_start+copy_h, x_start:x_start+copy_w] = arr[y_src_start:y_src_start+copy_h, x_src_start:x_src_start+copy_w]

    return result
