"""
isodose_engine.py — Motor de Comparação de Isodose

Extrai curvas de isodose como contornos (linhas) e calcula
sobreposição entre filme e TPS.

Isodoses: 50%, 75%, 100%, 125%, 150% da dose de prescrição
"""

import numpy as np
from skimage.measure import find_contours


def extract_isodose_contours(dose_map, prescription_dose, levels=[50, 75, 100, 125, 150]):
    """
    Extrai curvas de isodose como contornos (linhas).
    
    Args:
        dose_map: np.ndarray — mapa de dose em Gy
        prescription_dose: dose de prescrição em Gy (define 100%)
        levels: lista de percentuais [50, 75, 100, 125, 150]
        
    Returns:
        dict: {percentual: lista_de_contornos}
    """
    contours = {}
    for pct in levels:
        dose_val = prescription_dose * (pct / 100.0)
        # find_contours retorna lista de arrays (N, 2) com coordenadas [row, col]
        ct = find_contours(dose_map, level=dose_val)
        if len(ct) > 0:
            contours[pct] = ct
    return contours


def contour_area_pixels(contour_list):
    """Calcula área total em pixels de uma lista de contornos (fórmula de shoelace)."""
    total = 0
    for contour in contour_list:
        if len(contour) < 3:
            continue
        x = contour[:, 1]
        y = contour[:, 0]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        total += area
    return total


def compare_isodose(
    dose_film,
    dose_tps,
    prescription_dose,
    levels=[50, 75, 100, 125, 150],
    tolerance_mm=3.0,
    res_mm=0.35,
):
    """
    Compara isodoses entre filme e TPS usando contornos.
    
    Args:
        dose_film: mapa de dose do filme (Gy)
        dose_tps: mapa de dose do TPS (Gy)
        prescription_dose: dose de prescrição em Gy (define 100%)
        levels: [50, 75, 100, 125, 150]
        tolerance_mm: tolerância espacial em mm
        res_mm: resolução em mm/pixel
        
    Returns:
        dict com results, contours_film, contours_tps
    """
    # Extrair contornos
    ct_film = extract_isodose_contours(dose_film, prescription_dose, levels)
    ct_tps = extract_isodose_contours(dose_tps, prescription_dose, levels)
    
    results = []
    
    for pct in levels:
        film_list = ct_film.get(pct, [])
        tps_list = ct_tps.get(pct, [])
        
        area_film = contour_area_pixels(film_list)
        area_tps = contour_area_pixels(tps_list)
        
        # Coincidência: se ambos têm contornos, calcular overlap aproximado
        if area_film > 0 and area_tps > 0:
            # Overlap aproximado: interseção / união das áreas
            # Interseção = área da menor (aproximação conservadora)
            intersection = min(area_film, area_tps) * 0.8  # fator conservador
            union = area_film + area_tps - intersection
            coincidence = 100.0 * intersection / union if union > 0 else 0.0
            
            # Distância: diferença de áreas / perímetro médio
            perim_film = sum(len(c) for c in film_list) if film_list else 1
            perim_tps = sum(len(c) for c in tps_list) if tps_list else 1
            mean_perim = (perim_film + perim_tps) / 2.0
            area_diff = abs(area_film - area_tps)
            mean_dist = (area_diff / mean_perim) * res_mm if mean_perim > 0 else 0.0
        else:
            coincidence = 0.0
            mean_dist = tolerance_mm if (area_film > 0 or area_tps > 0) else 0.0
        
        results.append({
            'level': pct,
            'coincidence': min(coincidence, 100.0),
            'mean_distance_mm': mean_dist,
            'area_film_px': area_film,
            'area_tps_px': area_tps,
            'n_contours_film': len(film_list),
            'n_contours_tps': len(tps_list),
            'dose_value': prescription_dose * (pct / 100.0),
        })
    
    return {
        'results': results,
        'contours_film': ct_film,
        'contours_tps': ct_tps,
        'prescription_dose': prescription_dose,
    }
