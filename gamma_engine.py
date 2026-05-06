"""
gamma_engine.py — Motor de Análise Gamma 2D (versão final e correta).

Baseado em:
  - Low et al. 1998 (fórmula do gamma index)
  - AAPM TG-119 / TG-218 (ROI, threshold, critérios)
  - OmniPro / FilmQA Pro (padrão comercial: TPS interpolado para filme, ROI delimitada)

Fluxo correto:
  1. TPS (baixa resolução) → interpolado para mesma resolução do filme
  2. TPS interpolado → redimensionado para MESMO SHAPE do filme
  3. Registro: testa 4 orientações do filme (normal, flipH, flipV, flipHV)
  4. ROI: define região de interesse (automática ou manual)
  5. Gamma: calcula apenas dentro da ROI
"""

import numpy as np
from scipy.ndimage import zoom, shift
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def resize_to_shape(source, target_shape):
    """Redimensiona 'source' para 'target_shape' via interpolação bilinear."""
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
    
    Estratégia:
      1. Normaliza para 0-1
      2. Threshold de Otsu para separar filme de fundo
      3. Seleciona maior componente conexo
      4. Adiciona margem de 5%
    
    Retorna:
      roi_mask: máscara binária (1 = dentro da ROI, 0 = fora)
      bbox: (x, y, w, h) bounding box da ROI
    """
    img = np.asarray(dose_map, dtype=np.float64)
    
    # Normalizar
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img)
    
    # Threshold de Otsu
    try:
        thresh = threshold_otsu(img_norm)
    except:
        thresh = 0.3
    
    # O filme irradiado tem dose ALTA = pixels CLAROS no mapa de dose
    mask = img_norm > thresh
    
    # Componentes conexos
    labeled = label(mask)
    regions = regionprops(labeled)
    
    if not regions:
        # Fallback: usar threshold percentual da dose máxima
        dose_thresh = (threshold_percent / 100.0) * img_max
        mask = img > dose_thresh
        labeled = label(mask)
        regions = regionprops(labeled)
    
    if regions:
        # Selecionar maior região
        largest = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest.bbox
        
        # Adicionar margem de 5%
        margin_y = int((maxr - minr) * 0.05)
        margin_x = int((maxc - minc) * 0.05)
        
        minr = max(0, minr - margin_y)
        minc = max(0, minc - margin_x)
        maxr = min(img.shape[0], maxr + margin_y)
        maxc = min(img.shape[1], maxc + margin_x)
        
        roi_mask = np.zeros_like(img, dtype=bool)
        roi_mask[minr:maxr, minc:maxc] = True
        
        return roi_mask, (minc, minr, maxc - minc, maxr - minr)
    
    # Fallback total: usar toda a imagem
    return np.ones_like(img, dtype=bool), (0, 0, img.shape[1], img.shape[0])


def register_with_flip(dose_film, dose_tps, res_mm):
    """
    Registro automático que testa 4 orientações do filme.
    
    Testa:
      1. Filme original
      2. Filme flip horizontal (fliplr)
      3. Filme flip vertical (flipud)
      4. Filme flip H + V
    
    Retorna:
      best_film: filme na melhor orientação
      best_flip: 'none', 'h', 'v', 'hv'
      tx, ty: translação ótima em mm
      score: score de correlação (0-1)
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
        # Cross-correlation 2D
        corr = correlate2d(dose_tps, film_test, mode='same', boundary='fill')
        
        # Encontrar pico de correlação
        max_idx = np.unravel_index(np.argmax(corr), corr.shape)
        cy, cx = np.array(film_test.shape) // 2
        shift_y = max_idx[0] - cy
        shift_x = max_idx[1] - cx
        
        tx_mm = float(shift_x * res_mm)
        ty_mm = float(shift_y * res_mm)
        
        # Score normalizado
        score = float(corr[max_idx] / (np.sum(dose_tps**2) * np.sum(film_test**2))**0.5)
        
        if score > best_score:
            best_score = score
            best_flip = flip_name
            best_tx = tx_mm
            best_ty = ty_mm
            best_film = film_test
    
    return best_film, best_flip, best_tx, best_ty, best_score


def gamma_analysis(
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
    Calcula gamma index 2D entre filme e TPS.
    
    PRÉ-CONDIÇÃO: dose_film e dose_tps devem ter o MESMO shape.
    
    Args:
        dose_film: np.ndarray — dose do filme (Gy)
        dose_tps: np.ndarray — dose do TPS alinhado (Gy), mesmo shape do filme
        res_mm: resolução em mm/pixel (ambos na mesma resolução)
        roi_mask: máscara binária opcional — só calcula gamma onde mask=True
        dose_percent: critério de dose (%)
        dta_mm: critério DTA (mm)
        threshold_percent: pixels com dose < threshold% do máx são ignorados
        global_norm: True = DD é % do máx global do TPS
        max_gamma: cap máximo
    
    Returns:
        dict com gamma_map, passing_rate, estatísticas
    """
    ref = np.asarray(dose_tps, dtype=np.float64)
    eval_ = np.asarray(dose_film, dtype=np.float64)
    
    assert ref.shape == eval_.shape, f"Shapes diferentes: TPS {ref.shape} vs Filme {eval_.shape}"
    
    h, w = ref.shape
    
    ref_max = np.max(ref)
    if ref_max <= 0:
        raise ValueError("Dose máxima do TPS é zero")
    
    # Threshold de dose (ROI baseada em dose)
    dose_threshold = ref_max * (threshold_percent / 100.0)
    
    # Dose difference absoluto
    dd_abs = ref_max * (dose_percent / 100.0) if global_norm else None
    
    # Search radius em pixels
    search_px = int(np.ceil(dta_mm / res_mm))
    
    gamma_map = np.full_like(ref, np.nan)
    
    for iy in range(h):
        for ix in range(w):
            dose_ref = ref[iy, ix]
            
            # Verificar ROI
            if roi_mask is not None and not roi_mask[iy, ix]:
                gamma_map[iy, ix] = -1.0
                continue
            
            # Verificar threshold de dose
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
            
            # Janela de busca no filme
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
