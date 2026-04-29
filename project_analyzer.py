import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import json
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, square, erosion, dilation
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

st.set_page_config(page_title="Project Analyzer", page_icon="🔬", layout="wide")

def mm_to_pixels(mm, dpi):
    return int((mm / 25.4) * dpi)

def calcular_roi_quadrado(largura_px, altura_px, dpi):
    px_por_cm = dpi / 2.54
    largura_cm = largura_px / px_por_cm
    altura_cm = altura_px / px_por_cm
    menor_dimensao_cm = min(largura_cm, altura_cm)
    roi_cm = menor_dimensao_cm * 0.60
    roi_cm = max(0.5, min(roi_cm, 2.5))
    roi_px = int(roi_cm * px_por_cm)
    return roi_px, roi_cm

def cortar_filme_unico(imagem):
    gray = para_grayscale(imagem)
    # Normalizar para 0-1 para Otsu funcionar corretamente com qualquer bit-depth
    if gray.max() > 1.0:
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min())
    else:
        gray_norm = gray
    thresh = threshold_otsu(gray_norm)
    binary = gray_norm < thresh
    binary = clear_border(binary)
    labeled = label(binary)
    regions = regionprops(labeled)
    if not regions:
        return None
    largest = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox
    margem = 10
    h, w = imagem.shape[:2]
    return imagem[max(0,minr-margem):min(h,maxr+margem), max(0,minc-margem):min(w,maxc+margem)]

def detectar_regioes_unico(imagem, area_min, offset, fechamento, erosao):
    gray = para_grayscale(imagem)
    if gray.max() > 1.0:
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min())
    else:
        gray_norm = gray
    thresh = threshold_otsu(gray_norm)
    thresh_ajustado = thresh * (1 - offset)
    binary = gray_norm < thresh_ajustado
    if erosao > 0:
        binary = erosion(binary, square(erosao))
    binary = remove_small_objects(binary, min_size=area_min)
    if fechamento > 0:
        binary = closing(binary, square(fechamento))
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray_norm)
    regioes = []
    for i, r in enumerate(regions):
        if r.area >= area_min:
            minr, minc, maxr, maxc = r.bbox
            w = maxc - minc
            h = maxr - minr
            razao = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            regioes.append({
                'idx': i, 'area': r.area, 'intensidade': r.mean_intensity,
                'centro': (int(r.centroid[1]), int(r.centroid[0])),
                'bbox': (minc, minr, w, h), 'razao': razao
            })
    return regioes, gray_norm

def detectar_filmes_multiplos(imagem, area_min):
    gray = para_grayscale(imagem)
    if gray.max() > 1.0:
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min())
    else:
        gray_norm = gray
    thresh = threshold_otsu(gray_norm)
    binary = gray_norm < thresh
    binary = clear_border(binary)
    binary = remove_small_objects(binary, min_size=area_min)
    binary = closing(binary, square(5))
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray_norm)
    filmes = []
    for i, r in enumerate(regions):
        if r.area >= area_min:
            minr, minc, maxr, maxc = r.bbox
            w = maxc - minc
            h = maxr - minr
            filme_cortado = imagem[minr:maxr, minc:maxc]
            filmes.append({
                'idx': i, 'imagem': filme_cortado, 'area': r.area,
                'intensidade_media': r.mean_intensity,
                'centro': (int(r.centroid[1]), int(r.centroid[0])),
                'bbox': (minc, minr, w, h), 'arquivo': ''
            })
    return filmes, binary

def ordenar(regioes):
    ordenadas = sorted(regioes, key=lambda x: x['intensidade_media'] if 'intensidade_media' in x else x['intensidade'], reverse=True)
    for i, r in enumerate(ordenadas, 1):
        r['id'] = i
    return ordenadas

def calcular_intensidade_roi(imagem_filme, roi_px):
    h, w = imagem_filme.shape[:2]
    cx = w // 2
    cy = h // 2
    meio_roi = roi_px // 2
    x1 = max(0, cx - meio_roi)
    y1 = max(0, cy - meio_roi)
    x2 = min(w, cx + meio_roi)
    y2 = min(h, cy + meio_roi)
    # Canal vermelho para dosimetria EBT (AAPM TG-55)
    red = canal_vermelho(imagem_filme)
    roi_pixels = red[y1:y2, x1:x2]
    intensidade_roi = float(np.mean(roi_pixels))
    intensidade_total = float(np.mean(red))
    bbox_roi = (x1, y1, x2 - x1, y2 - y1)
    return intensidade_roi, bbox_roi, intensidade_total

def desenhar_marcacoes_original(imagem, filmes, dpi, mostrar_recuo=True, mostrar_roi=True):
    # Normalizar para uint8 apenas para display
    img = normalizar_para_display(imagem)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        fonte = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
    except:
        fonte = ImageFont.load_default()
    recuo_px = mm_to_pixels(5, dpi)
    for f in filmes:
        x, y, w, h = f['bbox']
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
        if mostrar_recuo:
            x_recuo = x + recuo_px
            y_recuo = y + recuo_px
            w_recuo = w - 2 * recuo_px
            h_recuo = h - 2 * recuo_px
            if w_recuo > 0 and h_recuo > 0:
                desenhar_tracejado_fino(draw, x_recuo, y_recuo, x_recuo + w_recuo, y_recuo + h_recuo, (255, 0, 0), 2)
        if mostrar_roi and 'roi_bbox' in f:
            rx, ry, rw, rh = f['roi_bbox']
            rx_abs = x + rx
            ry_abs = y + ry
            draw.rectangle([rx_abs, ry_abs, rx_abs + rw, ry_abs + rh], outline=(0, 102, 255), width=3)
        cx, cy = f['centro']
        txt = str(f['id'])
        bbox = draw.textbbox((0, 0), txt, font=fonte)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.rectangle([cx-tw//2-4, cy-th//2-4, cx+tw//2+4, cy+th//2+4], fill=(0,0,0))
        draw.text((cx-tw//2, cy-th//2), txt, fill=(255,255,255), font=fonte)
    return np.array(img_pil)

def desenhar_marcacoes_filme(imagem_filme, roi_bbox, recuo_px, dpi):
    img = normalizar_para_display(imagem_filme)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    h, w = img.shape[:2]
    draw.rectangle([0, 0, w-1, h-1], outline=(0, 255, 0), width=2)
    if recuo_px > 0:
        x_recuo = recuo_px
        y_recuo = recuo_px
        w_recuo = w - 2 * recuo_px
        h_recuo = h - 2 * recuo_px
        if w_recuo > 0 and h_recuo > 0:
            desenhar_tracejado_fino(draw, x_recuo, y_recuo, x_recuo + w_recuo, y_recuo + h_recuo, (255, 0, 0), 1)
    if roi_bbox:
        rx, ry, rw, rh = roi_bbox
        draw.rectangle([rx, ry, rx + rw, ry + rh], outline=(0, 102, 255), width=2)
    return np.array(img_pil)

def desenhar_tracejado_fino(draw, x1, y1, x2, y2, cor, largura, segmento=5, espaco=3):
    i = x1
    while i < x2:
        seg = min(segmento, x2 - i)
        draw.line([(i, y1), (i + seg, y1)], fill=cor, width=largura)
        i += segmento + espaco
    i = x1
    while i < x2:
        seg = min(segmento, x2 - i)
        draw.line([(i, y2), (i + seg, y2)], fill=cor, width=largura)
        i += segmento + espaco
    i = y1
    while i < y2:
        seg = min(segmento, y2 - i)
        draw.line([(x1, i), (x1, i + seg)], fill=cor, width=largura)
        i += segmento + espaco
    i = y1
    while i < y2:
        seg = min(segmento, y2 - i)
        draw.line([(x2, i), (x2, i + seg)], fill=cor, width=largura)
        i += segmento + espaco

def para_grayscale(imagem):
    """Converte para grayscale mantendo range original (para deteccao)"""
    if len(imagem.shape) == 3 and imagem.shape[2] >= 3:
        # Usar canal vermelho para deteccao tambem (melhor para filmes EBT)
        return imagem[..., 0].astype(np.float64)
    return imagem.astype(np.float64)

def canal_vermelho(imagem):
    """Extrai canal vermelho para dosimetria EBT (AAPM TG-55)"""
    if len(imagem.shape) == 3 and imagem.shape[2] >= 3:
        return imagem[..., 0].astype(np.float64)
    return imagem.astype(np.float64)

def normalizar_para_display(imagem):
    """Normaliza imagem para uint8 apenas para visualizacao"""
    img_float = imagem.astype(np.float64)
    img_min = img_float.min()
    img_max = img_float.max()
    if img_max > img_min:
        img_norm = (img_float - img_min) / (img_max - img_min) * 255.0
    else:
        img_norm = img_float
    return img_norm.clip(0, 255).astype(np.uint8)

def carregar_imagem_preservando_bits(arquivo):
    """
    Carrega imagem TIFF do scanner preservando bit-depth original.
    Baseado em Dosepy (imageio.v3) e cobaltCorsair (tifffile).
    """
    arquivo.seek(0)
    img_array = tifffile.imread(io.BytesIO(arquivo.read()))
    
    info = {
        'dtype': str(img_array.dtype),
        'shape': img_array.shape,
        'max_val': float(img_array.max()) if img_array.size > 0 else 0,
        'min_val': float(img_array.min()) if img_array.size > 0 else 0
    }
    
    # TIFFs de 16-bit scanner geralmente sao uint16 com shape (H, W, 3) para RGB
    # Se for float com max <= 1.0, provavelmente ja esta normalizado (escalar de volta)
    if img_array.dtype in [np.float32, np.float64] and img_array.max() <= 1.0:
        img_array = (img_array * 65535.0).clip(0, 65535).astype(np.uint16)
        info['dtype'] = 'uint16 (escalado de float)'
        info['max_val'] = float(img_array.max())
    
    # Garantir que eh array numpy
    if not isinstance(img_array, np.ndarray):
        raise TypeError(f"Formato de imagem nao suportado: {type(img_array)}")
    
    return img_array, info

def ajustar_bbox(bbox, encolher, expandir):
    x, y, w, h = bbox
    if encolher > 0:
        x += encolher; y += encolher; w -= 2 * encolher; h -= 2 * encolher
    elif expandir > 0:
        x -= expandir; y -= expandir; w += 2 * expandir; h += 2 * expandir
    w = max(10, w); h = max(10, h)
    return (x, y, w, h)

# ==================== FUNCOES DE CALIBRACAO ====================

def calcular_nod(filmes_calibracao):
    """
    Calcula NOD corretamente considerando a direcao do scanner.
    
    EBT3/EBT4: NOD = log10(PV0 / PVirradiado) para transmissao padrao
    Se scanner inverte (ADC cresce com dose): NOD = log10(PVirradiado / PV0)
    
    Usa o filme de dose=0 como PV0, ou filme marcado como 'filme0' (upload separado).
    """
    # Sempre ordenar por dose (usado depois para filme_max tambem)
    filmes_ordenados_por_dose = sorted(filmes_calibracao, key=lambda f: f['dose'])
    
    # PRIORIDADE 1: filme marcado como 'filme0' (upload separado de filme 0 Gy)
    filme_0_especial = None
    for f in filmes_calibracao:
        if f.get('filme', {}).get('filme0', False):
            filme_0_especial = f
            break
    
    # PRIORIDADE 2: filme de menor dose (= 0 Gy)
    if filme_0_especial is not None:
        filme_0 = filme_0_especial
        pv0 = float(filme_0['filme']['intensidade_roi'])
        dose_0 = 0.0  # Forcar dose zero para upload separado
    else:
        filme_0 = filmes_ordenados_por_dose[0]
        pv0 = float(filme_0['filme']['intensidade_roi'])
        dose_0 = float(filme_0['dose'])
    
    # VALIDACAO CRITICA: o filme de referencia DEVE ter dose ~0 Gy
    # Se nao houver filme de 0 Gy, o NOD nao pode ser calculado corretamente
    if dose_0 > 0.001:
        info = {
            'pv0': pv0,
            'dose_0': dose_0,
            'pv_max': 0,
            'dose_max': 0,
            'adc_aumenta_com_dose': False,
            'erro': True,
            'erro_msg': f"NAO HA FILME DE 0 Gy! O filme de menor dose tem {dose_0:.2f} Gy. E obrigatorio incluir um filme NAO IRRADIADO (0 Gy) como referencia."
        }
        # Marcar todos com erro
        for f in filmes_calibracao:
            f['nod'] = 0.0
            f['nod_info'] = 'ERRO: Sem filme de 0 Gy na calibracao'
        return pv0, info
    
    # Encontrar filme de maior dose
    filme_max = filmes_ordenados_por_dose[-1]
    pv_max = float(filme_max['filme']['intensidade_roi'])
    dose_max = float(filme_max['dose'])
    
    # Detectar direcao: ADC aumenta ou diminui com dose?
    adc_aumenta_com_dose = pv_max > pv0
    
    info = {
        'pv0': pv0,
        'dose_0': dose_0,
        'pv_max': pv_max,
        'dose_max': dose_max,
        'adc_aumenta_com_dose': adc_aumenta_com_dose,
        'erro': False,
        'erro_msg': ''
    }
    
    for f in filmes_calibracao:
        pv_irrad = float(f['filme']['intensidade_roi'])
        
        if pv0 <= 0 or pv_irrad <= 0:
            f['nod'] = 0.0
            f['nod_info'] = 'Erro: PV <= 0'
            continue
            
        if adc_aumenta_com_dose:
            # Scanner tipo IBA: escuro = maior valor (refletancia/invertido)
            # NOD = log10(PVirradiado / PV0)
            f['nod'] = float(np.log10(pv_irrad / pv0))
            f['nod_info'] = f'log10({pv_irrad:.1f}/{pv0:.1f}) = {f["nod"]:.4f}'
        else:
            # Scanner de transmissao padrao: escuro = menor valor
            # NOD = log10(PV0 / PVirradiado)
            f['nod'] = float(np.log10(pv0 / pv_irrad))
            f['nod_info'] = f'log10({pv0:.1f}/{pv_irrad:.1f}) = {f["nod"]:.4f}'
    
    return pv0, info

def fitting_polinomial2(nods, doses):
    """Fitting: Dose = a*NOD^2 + b*NOD + c"""
    coefs = np.polyfit(nods, doses, 2)
    a, b, c = coefs
    
    doses_pred = a * nods**2 + b * nods + c
    ss_res = np.sum((doses - doses_pred)**2)
    ss_tot = np.sum((doses - np.mean(doses))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {'a': a, 'b': b, 'c': c, 'r2': r2, 'equation': f"Dose = {a:.4f}*NOD² + {b:.4f}*NOD + {c:.4f}"}

def fitting_potencia(nods, doses):
    """Fitting: Dose = K1 * NOD^K2"""
    from scipy.optimize import curve_fit
    
    def func_potencia(x, K1, K2):
        return K1 * (x ** K2)
    
    try:
        popt, _ = curve_fit(func_potencia, nods, doses, p0=[1.0, 2.0])
        K1, K2 = popt
        
        doses_pred = K1 * (nods ** K2)
        ss_res = np.sum((doses - doses_pred)**2)
        ss_tot = np.sum((doses - np.mean(doses))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {'K1': K1, 'K2': K2, 'r2': r2, 'equation': f"Dose = {K1:.4f} * NOD^{K2:.4f}"}
    except Exception as e:
        st.error(f"Erro no fitting: {e}")
        return None

def fitting_polinomial_n(nods, doses):
    """
    Fitting polinomial de grau n: Dose = a*NOD + b*NOD^n
    Usado por Dosepy (polynomial_n).
    Mais flexivel que polinomial 2o grau porque o expoente n eh ajustado.
    """
    from scipy.optimize import curve_fit
    
    def func_poly_n(x, a, b, n):
        return a * x + b * (x ** n)
    
    try:
        # p0 baseado em dados tipicos de EBT3/EBT4
        # a: coeficiente linear (~5-15), b: coeficiente da potencia (~20-50), n: expoente (~1.5-3.0)
        popt, _ = curve_fit(
            func_poly_n, nods, doses, 
            p0=[5.0, 30.0, 2.0],
            bounds=([0, 0, 1.0], [100, 200, 5.0]),
            maxfev=5000
        )
        a, b, n = popt
        
        doses_pred = a * nods + b * (nods ** n)
        ss_res = np.sum((doses - doses_pred)**2)
        ss_tot = np.sum((doses - np.mean(doses))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {'a': a, 'b': b, 'n': n, 'r2': r2, 'equation': f"Dose = {a:.4f}*NOD + {b:.4f}*NOD^{n:.4f}", 'type': 'polynomial_n'}
    except Exception as e:
        st.error(f"Erro no fitting polinomial_n: {e}")
        return None

def fitting_racional(nods, doses):
    """
    Fitting racional: Dose = -c + b/(NOD - a)
    Usado por Dosepy (rational_function) e cobaltCorsair (fit_func).
    Muito eficaz para EBT3. Pode falhar se NOD estiver muito proximo de 'a'.
    """
    from scipy.optimize import curve_fit
    
    def func_racional(x, a, b, c):
        return -c + b / (x - a)
    
    try:
        # Para evitar divisao por zero, 'a' deve ser negativo (menor que qualquer NOD)
        popt, _ = curve_fit(
            func_racional, nods, doses, 
            p0=[-0.5, 50.0, 5.0],
            bounds=([-10.0, 0.1, 0.1], [-0.001, 500.0, 50.0]),
            maxfev=5000
        )
        a, b, c = popt
        
        doses_pred = -c + b / (nods - a)
        ss_res = np.sum((doses - doses_pred)**2)
        ss_tot = np.sum((doses - np.mean(doses))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {'a': a, 'b': b, 'c': c, 'r2': r2, 'equation': f"Dose = -{c:.4f} + {b:.4f}/(NOD - {a:.4f})", 'type': 'racional'}
    except Exception as e:
        st.warning(f"Fitting racional falhou: {e}. Tente outro tipo de fitting.")
        return None

def _calcular_dose_curva(nod, curva):
    """Calcula dose predita a partir de um NOD usando a curva ajustada."""
    if curva.get('type') == 'racional':
        return -curva['c'] + curva['b'] / (nod - curva['a'])
    elif curva.get('type') == 'polynomial_n':
        return curva['a'] * nod + curva['b'] * (nod ** curva['n'])
    elif 'K1' in curva:
        return curva['K1'] * (nod ** curva['K2'])
    else:
        return curva['a'] * nod**2 + curva['b'] * nod + curva['c']

def gerar_grafico_nod_dose(filmes, curva, tipo_filme):
    """Grafico NOD vs Dose (padrao cientifico)"""
    nods = np.array([f['nod'] for f in filmes])
    doses = np.array([f['dose'] for f in filmes])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(nods, doses, color='red', s=150, label='Dados medidos', zorder=5, edgecolors='black', linewidth=1)
    
    nods_fit = np.linspace(0, max(nods)*1.15, 200)
    doses_fit = [_calcular_dose_curva(n, curva) for n in nods_fit]
    
    ax.plot(nods_fit, doses_fit, 'b-', linewidth=2.5, label='Curva ajustada', zorder=3)
    
    for i in range(len(nods)):
        dose_na_curva = _calcular_dose_curva(nods[i], curva)
        ax.plot([nods[i], nods[i]], [doses[i], dose_na_curva], 'g--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('NOD (Net Optical Density)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dose (Gy)', fontsize=14, fontweight='bold')
    ax.set_title(f'Curva de Calibração {tipo_filme} - NOD vs Dose\n{curva["equation"]}\nR² = {curva["r2"]:.6f}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def gerar_grafico_adc_dose(filmes, tipo_filme):
    """
    Grafico ADC (Pixel Value) vs Dose - relacao direta.
    NUNCA faz fitting polinomial em ADC (numericamente instavel com valores 10k-60k).
    Mostra apenas pontos + spline suave para visualizacao.
    """
    adcs = np.array([f['filme']['intensidade_roi'] for f in filmes])
    doses = np.array([f['dose'] for f in filmes])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(adcs, doses, color='darkorange', s=150, label='Dados medidos', zorder=5, edgecolors='black', linewidth=1)
    
    # Spline suave apenas para visualizacao (nao eh fitting!)
    try:
        from scipy.interpolate import make_interp_spline
        # Ordenar por ADC para spline fazer sentido
        ordem = np.argsort(adcs)
        adcs_ord = adcs[ordem]
        doses_ord = doses[ordem]
        
        # Só faz spline se tivermos pelo menos 4 pontos
        if len(adcs) >= 4:
            x_spline = np.linspace(min(adcs), max(adcs), 200)
            spl = make_interp_spline(adcs_ord, doses_ord, k=2)
            y_spline = spl(x_spline)
            # Clip para nao ir abaixo de zero
            y_spline = np.clip(y_spline, 0, None)
            ax.plot(x_spline, y_spline, 'purple', linewidth=2.0, alpha=0.6, linestyle='--', label='Interpolacao (visual)', zorder=2)
    except Exception:
        pass
    
    ax.set_xlabel('ADC (Average Digitized Count)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dose (Gy)', fontsize=14, fontweight='bold')
    ax.set_title(f'Curva de Calibração {tipo_filme} - ADC vs Dose\n(Canal Vermelho, valores brutos do scanner)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylim(bottom=0)
    
    # Inverter eixo X apenas se ADC diminui com dose (transmissao padrao)
    if len(adcs) >= 2 and doses[-1] > doses[0]:
        if adcs[-1] < adcs[0]:
            ax.invert_xaxis()
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ==================== INTERFACE ====================

st.title("🔬 Project Analyzer v9.1")
st.markdown("**Corrigido:** tifffile para 16-bit | Fittings Dosepy (polynomial_n, racional) | ADC sem fitting instável")
st.info("ℹ️ O app agora usa apenas o **canal vermelho** e preserva o **bit-depth original** do scanner. Valores de ADC devem estar na faixa de milhares (ex: 27000-52000 para 16-bit).")

tipo_filme = st.radio("Qual filme voce esta analisando?", ["EBT3", "EBT4"], horizontal=True)
metodologia = st.radio("Qual a metodologia?", ["Um unico filme", "Varios filmes"], horizontal=True)
st.markdown("---")

with st.sidebar:
    st.header("Configuracoes")
    dpi = st.number_input("DPI do Scanner", 1, 2400, 50)
    if metodologia == "Um unico filme":
        st.subheader("Modo Unico Filme")
        area_min = st.slider("Area Minima", 100, 50000, 3000, 100)
        offset = st.slider("Sensibilidade", 0.0, 0.5, 0.15, 0.01)
        fechamento = st.slider("Fechamento", 0, 20, 5, 1)
        erosao_global = st.slider("Erosao Global", 0, 10, 0, 1)
    else:
        st.subheader("Modo Varios Filmes")
        area_min_multi = st.slider("Area Minima por Filme", 100, 50000, 2000, 100)
        mostrar_recuo = st.checkbox("Mostrar recuo 5mm", value=True)
        mostrar_roi = st.checkbox("Mostrar ROI", value=True)
        
        st.markdown("---")
        st.subheader("Scanner / ADC")
        inverter_adc = st.checkbox("Scanner com sinal invertido (reflectancia / ADC cresce com dose)", value=False,
                                   help="Marque se seu scanner gera valores que AUMENTAM com a dose (como no software IBA). Scanners de transmissao tipicamente geram valores que DIMINUEM com a dose.")
        valor_max_scanner = st.number_input("Valor maximo do scanner (bit-depth)", value=65535, min_value=255, max_value=262143, step=1,
                                            help="65535 = 16-bit | 16383 = 14-bit | 4095 = 12-bit | 255 = 8-bit")
        
        st.markdown("---")
        st.subheader("Curva de Calibração")
        if 'curva_calibracao' in st.session_state:
            st.success("✅ Curva salva na sessão")
            st.info(f"Tipo: {st.session_state['curva_calibracao']['tipo_filme']}")
            st.info(f"R²: {st.session_state['curva_calibracao']['r2']:.4f}")
        else:
            st.warning("Nenhuma curva salva")
        
        curva_upload = st.file_uploader("Carregar curva salva (.json)", type=['json'], key="curva_upload")
        if curva_upload:
            try:
                curva_data = json.load(io.BytesIO(curva_upload.read()))
                st.session_state['curva_calibracao'] = curva_data
                st.success("Curva carregada com sucesso!")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao carregar: {e}")

# ==================== MODO UNICO FILME ====================

if metodologia == "Um unico filme":
    st.header("Upload do Filme Irradiado")
    arquivo = st.file_uploader("Envie a imagem do filme irradiado", type=['tif','tiff','png','jpg','jpeg'])
    
    if arquivo:
        img_orig, img_info = carregar_imagem_preservando_bits(arquivo)
        
        # Mostrar info da imagem
        st.info(f"📷 Imagem: {img_info['dtype']} | shape{img_info['shape']} | range [{img_info['min_val']:.1f}, {img_info['max_val']:.1f}]")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(normalizar_para_display(img_orig), use_container_width=True)
        
        with col2:
            st.subheader("Canal Vermelho (EBT)")
            st.image(normalizar_para_display(canal_vermelho(img_orig).reshape(img_orig.shape[:2])), use_container_width=True)
        
        if st.button("ANALISAR", type="primary", key="btn_unico"):
            with st.spinner("Processando..."):
                img_filme = cortar_filme_unico(img_orig)
                if img_filme is None:
                    st.error("Filme nao detectado!")
                    st.stop()
                regioes, gray = detectar_regioes_unico(img_filme, area_min, offset, fechamento, erosao_global)
                if not regioes:
                    st.warning("Nenhuma regiao! Aumente Sensibilidade")
                    st.stop()
                st.session_state['regioes_unico'] = regioes
                st.session_state['img_filme_unico'] = img_filme
                st.session_state['dpi_unico'] = dpi
                st.rerun()
        
        if 'regioes_unico' in st.session_state:
            regioes = st.session_state['regioes_unico']
            img_filme = st.session_state['img_filme_unico']
            dpi_s = st.session_state['dpi_unico']
            
            st.markdown("---")
            st.header("Ajuste Individual das Regioes")
            regioes_ajust = []
            for i, regiao in enumerate(regioes):
                st.markdown(f"**Regiao {i+1}** (Intensidade: {regiao['intensidade']:.6f})")
                col_e, col_d = st.columns(2)
                with col_e:
                    enc = st.slider(f"Encolher R{i+1}", 0, 50, 0, 5, key=f"er_{i}")
                with col_d:
                    exp = st.slider(f"Expandir R{i+1}", 0, 50, 0, 5, key=f"di_{i}")
                bbox_ajust = ajustar_bbox(regiao['bbox'], enc, exp)
                regioes_ajust.append({
                    'id': i+1, 'area': regiao['area'], 'intensidade': regiao['intensidade'],
                    'centro': regiao['centro'], 'bbox': bbox_ajust, 'razao': regiao['razao']
                })
            reg_ord = ordenar(regioes_ajust)
            
            # Calcular ROI e intensidade do canal vermelho para cada regiao
            for r in reg_ord:
                roi_px, roi_cm = calcular_roi_quadrado(r['bbox'][2], r['bbox'][3], dpi_s)
                r['roi_px'] = roi_px
                r['roi_cm'] = roi_cm
                # Calcular intensidade do canal vermelho na ROI
                h, w = img_filme.shape[:2]
                cx = int(r['centro'][0])
                cy = int(r['centro'][1])
                meio_roi = roi_px // 2
                x1 = max(0, cx - meio_roi)
                y1 = max(0, cy - meio_roi)
                x2 = min(w, cx + meio_roi)
                y2 = min(h, cy + meio_roi)
                red = canal_vermelho(img_filme)
                roi_pixels = red[y1:y2, x1:x2]
                r['intensidade_roi'] = float(np.mean(roi_pixels))
            
            img_res = desenhar_marcacoes_original(img_filme, reg_ord, dpi_s)
            st.subheader("Resultado Final")
            st.image(img_res, use_container_width=True)
            cm = 1 / (dpi_s / 2.54)
            df = pd.DataFrame([{
                'Filme': r['id'], 'Area_cm2': round(r['area'] * cm * cm, 2),
                'Largura_cm': round(r['bbox'][2] * cm, 2),
                'Altura_cm': round(r['bbox'][3] * cm, 2),
                'ADC_Canal_R': round(r['intensidade_roi'], 1),
                'ROI_cm': round(r['roi_cm'], 2),
                'Razao': round(r['razao'], 2)
            } for r in reg_ord])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False), "resultado.csv", "text/csv")
            
            # Upload separado do filme de 0 Gy (nao irradiado / referencia)
            st.markdown("---")
            st.subheader("➕ Filme de Referência (0 Gy)")
            st.info("Se o filme de 0 Gy (nao irradiado) nao foi detectado automaticamente na imagem acima, faca o upload dele separadamente.")
            
            tem_filme_0 = st.checkbox("Fazer upload do filme de 0 Gy separado", value=False, key="chk_filme0")
            
            if tem_filme_0:
                arq_filme0 = st.file_uploader("Envie a imagem do filme de 0 Gy (nao irradiado)", type=['tif','tiff','png','jpg','jpeg'], key="upload_filme0")
                if arq_filme0:
                    img_f0, info_f0 = carregar_imagem_preservando_bits(arq_filme0)
                    st.info(f"Filme 0 Gy: {info_f0['dtype']} | shape{info_f0['shape']} | range [{info_f0['min_val']:.1f}, {info_f0['max_val']:.1f}]")
                    
                    # Detectar filme na imagem separada
                    filme0_cortado = cortar_filme_unico(img_f0)
                    if filme0_cortado is not None:
                        # Calcular ADC (canal vermelho) do filme 0 Gy
                        red_f0 = canal_vermelho(filme0_cortado)
                        adc_f0 = float(np.mean(red_f0))
                        
                        # Adicionar como regiao especial na lista
                        novo_id = max([r['id'] for r in reg_ord]) + 1 if reg_ord else 1
                        regiao_f0 = {
                            'id': novo_id,
                            'area': filme0_cortado.shape[0] * filme0_cortado.shape[1],
                            'intensidade': adc_f0,
                            'centro': (filme0_cortado.shape[1]//2, filme0_cortado.shape[0]//2),
                            'bbox': (0, 0, filme0_cortado.shape[1], filme0_cortado.shape[0]),
                            'razao': 1.0,
                            'roi_px': min(filme0_cortado.shape[0], filme0_cortado.shape[1]),
                            'roi_cm': min(filme0_cortado.shape[0], filme0_cortado.shape[1]) / (dpi_s / 2.54) if dpi_s > 0 else 0,
                            'intensidade_roi': adc_f0,
                            'filme0': True  # marcador especial
                        }
                        reg_ord.append(regiao_f0)
                        st.success(f"Filme 0 Gy adicionado! Regiao {novo_id} | ADC = {adc_f0:.1f}")
                        st.image(normalizar_para_display(filme0_cortado), caption=f"Filme 0 Gy detectado (ADC = {adc_f0:.1f})", use_container_width=True)
            
            # ==================== CURVA DE CALIBRACAO (MODO UNICO FILME) ====================
            st.markdown("---")
            st.header("📊 Curva de Calibração")
            
            st.error("🚨 **OBRIGATORIO: Voce DEVE incluir uma regiao com DOSE = 0 Gy (filme NAO IRRADIADO).** Sem o filme de referencia, o NOD nao pode ser calculado corretamente.")
            st.info("💡 **Dica:** Se o filme de 0 Gy nao apareceu na deteccao automatica (filme transparente), use a secao 'Filme de Referencia (0 Gy)' abaixo para fazer upload separado.")
            st.error("🚨 **IMPORTANTE: Use apenas arquivos TIF originais do scanner (16-bit / 48-bit color).** Screenshots, JPGs ou PNGs exportados perdem a precisão e geram resultados absurdos.")
            st.info(f"Tipo de filme selecionado: **{tipo_filme}**")
            st.info(f"Total de regioes detectadas: {len(reg_ord)}")
            st.warning("⚠️ Selecione pelo menos 3 regioes de calibração com doses conhecidas, incluindo UMA regiao de 0 Gy.")
            
            # Unidade
            unidade = st.radio("Unidade da dose", ["Gy", "cGy"], horizontal=True, key="uni_unico")
            
            # Tipo de fitting
            tipo_fitting = st.radio("Tipo de fitting", 
                ["Polinomial 2o grau", "Polinomial n (Dosepy)", "Racional (Dosepy/cobaltCorsair)", "Potencia"], 
                horizontal=True, key="fit_unico",
                help="Polinomial 2o: Dose=a*NOD²+b*NOD+c | Polinomial n: Dose=a*NOD+b*NOD^n (Dosepy) | Racional: Dose=-c+b/(NOD-a) (Dosepy/cobaltCorsair) | Potencia: Dose=K1*NOD^K2")
            
            # Modo manual de ADC
            usar_adc_manual = st.checkbox("🔧 Usar valores de ADC manualmente (ignorar leitura do scan)", value=False,
                                          key="adc_manual_unico",
                                          help="Se os valores lidos do TIFF parecem errados, marque esta opção e digite os ADCs manualmente para cada regiao.")
            if usar_adc_manual:
                st.info("Modo manual ativado. Digite os valores de ADC que você obteve do scanner/software de referência.")
            
            filmes_calibracao = []
            
            for i, regiao in enumerate(reg_ord):
                if usar_adc_manual:
                    col_check, col_info, col_adc, col_dose = st.columns([1, 2, 2, 2])
                else:
                    col_check, col_info, col_dose = st.columns([1, 3, 2])
                    
                with col_check:
                    usar = st.checkbox(f"Usar", key=f"calib_u_{i}")
                with col_info:
                    st.markdown(f"**Regiao {regiao['id']}** | ROI: {regiao['roi_cm']:.1f} cm")
                    if not usar_adc_manual:
                        st.caption(f"ADC auto: {regiao['intensidade_roi']:.1f}")
                
                if usar_adc_manual:
                    with col_adc:
                        adc_manual = st.number_input(f"ADC", min_value=0.0, value=float(regiao['intensidade_roi']), step=1.0, key=f"adc_manual_u_{i}")
                
                with col_dose:
                    dose_val = st.number_input(f"Dose (Gy/cGy)", min_value=0.0, value=0.0, step=0.1, key=f"dose_u_{i}")
                
                if usar:
                    filme_calib = {
                        'filme': dict(regiao),
                        'dose': dose_val,
                        'id': regiao['id']
                    }
                    if usar_adc_manual:
                        filme_calib['filme']['intensidade_roi'] = adc_manual
                    filmes_calibracao.append(filme_calib)
            
            # Botao gerar curva
            if st.button("🔬 GERAR CURVA DE CALIBRAÇÃO", type="primary", key="btn_curva_unico"):
                if len(filmes_calibracao) < 3:
                    st.error("⚠️ Selecione pelo menos 3 regioes para gerar a curva!")
                else:
                    with st.spinner("Calculando curva de calibração..."):
                        # Converter dose para Gy se necessario
                        if unidade == "cGy":
                            for f in filmes_calibracao:
                                f['dose'] = f['dose'] / 100.0
                        
                        # Calcular NOD para TODOS os filmes
                        pv0, nod_info = calcular_nod(filmes_calibracao)
                        
                        # VERIFICACAO CRITICA: existe filme de 0 Gy?
                        if nod_info.get('erro', False):
                            st.error("🚨 " + nod_info['erro_msg'])
                            st.error("⚠️ SOLUCAO: Marque a checkbox 'Usar' para um filme com Dose = 0 Gy (nao irradiado) e tente novamente.")
                            st.stop()
                        
                        nods = np.array([f['nod'] for f in filmes_calibracao])
                        doses = np.array([f['dose'] for f in filmes_calibracao])
                        adcs = np.array([f['filme']['intensidade_roi'] for f in filmes_calibracao])
                        
                        # Info do calculo
                        st.info(f"**Referência 0 Gy:** ADC = {pv0:.1f} | Dose = {nod_info['dose_0']:.2f} Gy")
                        st.info(f"**Maior dose:** ADC = {nod_info['pv_max']:.1f} | Dose = {nod_info['dose_max']:.2f} Gy")
                        
                        if nod_info['adc_aumenta_com_dose']:
                            st.success("✅ Scanner detectado: ADC **aumenta** com dose (tipo IBA/refletância). NOD = log10(PV/PV₀)")
                        else:
                            st.success("✅ Scanner detectado: ADC **diminui** com dose (transmissão padrão). NOD = log10(PV₀/PV)")
                        
                        # Debug table
                        debug_nods = pd.DataFrame([{
                            'Regiao': f['id'],
                            'Dose_Gy': f['dose'],
                            'ADC': f['filme']['intensidade_roi'],
                            'NOD': f['nod'],
                            'Forma': f.get('nod_info', '')
                        } for f in filmes_calibracao])
                        st.subheader("Debug — Cálculo de NOD")
                        st.dataframe(debug_nods, use_container_width=True, hide_index=True)
                        
                        # Verificar monotonicidade
                        for i in range(1, len(filmes_calibracao)):
                            if nods[i] < nods[i-1] and doses[i] > doses[i-1]:
                                st.warning(f"⚠️ NOD da regiao {filmes_calibracao[i]['id']} ({nods[i]:.4f}) < regiao anterior ({nods[i-1]:.4f}) apesar de dose maior.")
                        
                        # Escolher fitting
                        if tipo_fitting == "Polinomial 2o grau":
                            curva = fitting_polinomial2(nods, doses)
                        elif tipo_fitting == "Polinomial n (Dosepy)":
                            curva = fitting_polinomial_n(nods, doses)
                        elif tipo_fitting == "Racional (Dosepy/cobaltCorsair)":
                            curva = fitting_racional(nods, doses)
                        else:
                            curva = fitting_potencia(nods, doses)
                        
                        if curva:
                            # Calcular doses preditas
                            doses_pred = np.array([_calcular_dose_curva(n, curva) for n in nods])
                            
                            st.success(f"✅ Curva ajustada: {curva['equation']}")
                            st.info(f"R² = {curva['r2']:.6f}")
                            
                            # Graficos
                            col_g1, col_g2 = st.columns(2)
                            with col_g1:
                                st.subheader("NOD vs Dose")
                                grafico_nod = gerar_grafico_nod_dose(filmes_calibracao, curva, tipo_filme)
                                st.image(grafico_nod, use_container_width=True)
                            with col_g2:
                                st.subheader("ADC vs Dose")
                                grafico_adc = gerar_grafico_adc_dose(filmes_calibracao, tipo_filme)
                                st.image(grafico_adc, use_container_width=True)
                            
                            # Tabela de erros
                            st.subheader("Tabela de Erros")
                            erros_gy = doses_pred - doses
                            erros_pct = []
                            for i in range(len(doses)):
                                if doses[i] > 0.001:
                                    erros_pct.append(float(abs((doses_pred[i] - doses[i]) / doses[i] * 100)))
                                else:
                                    erros_pct.append(0.0)
                            
                            dados_tabela = []
                            for i in range(len(filmes_calibracao)):
                                dados_tabela.append({
                                    'Regiao': int(filmes_calibracao[i]['id']),
                                    'NOD': float(nods[i]),
                                    'ADC': float(adcs[i]),
                                    'Dose_Real_Gy': float(doses[i]),
                                    'Dose_Predita_Gy': float(doses_pred[i]),
                                    'Desvio_Gy': float(erros_gy[i]),
                                    'Erro_%': float(erros_pct[i])
                                })
                            
                            df_erros = pd.DataFrame(dados_tabela)
                            st.dataframe(df_erros, use_container_width=True, hide_index=True)
                            
                            # Download
                            st.download_button("Download CSV (erros)", df_erros.to_csv(index=False), "erros_calibracao.csv", "text/csv")
                            
                            # Salvar na sessao
                            resultado = {
                                'tipo_filme': tipo_filme,
                                'curva': curva,
                                'filmes': filmes_calibracao,
                                'df_erros': df_erros
                            }
                            st.session_state['curva_calibracao'] = resultado
                            st.success("✅ Curva salva na sessão! Vá para 'Vários filmes' para usar esta curva.")
                            
                            # Download JSON
                            curva_json = {
                                'tipo_filme': tipo_filme,
                                'curva': curva,
                                'tipo_fitting': tipo_fitting
                            }
                            st.download_button("📥 Download curva (.json)", json.dumps(curva_json, indent=2), "curva_calibracao.json", "application/json")
            
            # Mostrar curva salva se existir
            if 'curva_calibracao' in st.session_state:
                st.markdown("---")
                st.subheader("Curva Salva na Sessão")
                c = st.session_state['curva_calibracao']
                st.info(f"Tipo: {c['tipo_filme']} | Equação: {c['curva']['equation']} | R²: {c['curva']['r2']:.4f}")

# ==================== MODO VARIOS FILMES ====================

else:
    st.header("Upload dos Filmes Escaneados")
    st.info("Dica: Segure Ctrl e clique para selecionar multiplos arquivos")
    arquivos = st.file_uploader("Envie uma ou mais imagens com os filmes",
                               type=['tif','tiff','png','jpg','jpeg'],
                               accept_multiple_files=True)
    
    if arquivos and len(arquivos) > 0:
        st.success(f"{len(arquivos)} arquivo(s) carregado(s)")
        
        # Mostrar info das imagens
        for arq in arquivos:
            _, info = carregar_imagem_preservando_bits(arq)
            st.caption(f"📷 {arq.name}: {info['dtype']} | shape{info['shape']} | range [{info['min_val']:.1f}, {info['max_val']:.1f}]")
            arq.seek(0)
        
        if st.button("DETECTAR FILMES", type="primary"):
            with st.spinner("Detectando filmes..."):
                todos_filmes = []
                imagens_originais = []
                
                for idx_arq, arquivo in enumerate(arquivos):
                    img_np, img_info = carregar_imagem_preservando_bits(arquivo)
                    
                    filmes, binary = detectar_filmes_multiplos(img_np, area_min_multi)
                    
                    for f in filmes:
                        f['arquivo'] = arquivo.name
                        roi_px, roi_cm = calcular_roi_quadrado(f['bbox'][2], f['bbox'][3], dpi)
                        f['roi_px'] = roi_px
                        f['roi_cm'] = roi_cm
                        intens_roi, bbox_roi, intens_total = calcular_intensidade_roi(f['imagem'], roi_px)
                        # Aplicar inversao de sinal se necessario
                        if inverter_adc:
                            intens_roi = valor_max_scanner - intens_roi
                            intens_total = valor_max_scanner - intens_total
                        f['intensidade_roi'] = intens_roi
                        f['intensidade_total'] = intens_total
                        f['roi_bbox'] = bbox_roi
                        f['adc_raw'] = intens_roi  # Salvar valor original tambem
                    
                    filmes_temp = ordenar(filmes)
                    img_com_contornos = desenhar_marcacoes_original(
                        img_np, filmes_temp, dpi, mostrar_recuo, mostrar_roi
                    )
                    
                    imagens_originais.append({
                        'nome': arquivo.name,
                        'imagem': img_com_contornos,
                        'mascara': binary,
                        'filmes': filmes_temp
                    })
                    
                    todos_filmes.extend(filmes)
                
                todos_filmes = ordenar(todos_filmes)
                st.session_state['todos_filmes'] = todos_filmes
                st.session_state['imagens_originais'] = imagens_originais
                st.session_state['dpi_multi'] = dpi
                st.session_state['img_infos_multi'] = img_info
                st.rerun()
        
        if 'todos_filmes' in st.session_state:
            st.markdown("---")
            st.header("Resultado Final - Todos os Filmes")
            
            todos_filmes = st.session_state['todos_filmes']
            imagens_originais = st.session_state['imagens_originais']
            dpi_m = st.session_state['dpi_multi']
            px_por_cm = dpi_m / 2.54
            recuo_px = mm_to_pixels(5, dpi_m)
            
            st.success(f"Total: {len(todos_filmes)} filmes | Ordenados: 1=claro -> {len(todos_filmes)}=escuro")
            
            # Mostrar imagens originais com deteccao
            st.subheader("Imagens Originais com Deteccao")
            for img_info in imagens_originais:
                col_orig, col_masc = st.columns([3, 1])
                with col_orig:
                    st.markdown(f"**{img_info['nome']}** - Original com marcacoes")
                    st.image(img_info['imagem'], use_container_width=True)
                with col_masc:
                    st.markdown("**Mascara de deteccao**")
                    st.image((img_info['mascara'] * 255).astype(np.uint8), use_container_width=True)
                st.info(f"{len(img_info['filmes'])} filme(s) nesta imagem")
            
            st.markdown("---")
            
            # Legenda das cores
            st.markdown("""
            <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; border: 2px solid #00FF00; margin-right: 8px;"></div>
                    <span>Contorno do filme (verde)</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; border: 1px dashed #FF0000; margin-right: 8px;"></div>
                    <span>Recuo 5mm AAPM TG-55 (vermelho tracejado)</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 20px; border: 2px solid #0066FF; margin-right: 8px;"></div>
                    <span>ROI centralizado 60% (azul)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Grid com filmes individuais e marcacoes
            st.subheader("Filmes Individuais com Recuo e ROI")
            cols_por_linha = 5
            for i in range(0, len(todos_filmes), cols_por_linha):
                cols = st.columns(cols_por_linha)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(todos_filmes):
                        filme = todos_filmes[idx]
                        img_com_marcas = desenhar_marcacoes_filme(
                            filme['imagem'], filme.get('roi_bbox'), recuo_px, dpi_m
                        )
                        with col:
                            st.markdown(f"**Filme {filme['id']}**")
                            st.image(img_com_marcas, use_container_width=True)
                            st.caption(f"ROI: {filme['roi_cm']:.1f} cm | ADC (Canal R): {filme['intensidade_roi']:.1f}")
            
            # Tabela completa
            st.markdown("---")
            st.header("Tabela Completa")
            
            df = pd.DataFrame([{
                'Filme': f['id'], 'Arquivo': f['arquivo'],
                'Area_cm2': round(f['area'] / (px_por_cm ** 2), 2),
                'Largura_cm': round(f['bbox'][2] / px_por_cm, 2),
                'Altura_cm': round(f['bbox'][3] / px_por_cm, 2),
                'ROI_cm': round(f['roi_cm'], 2),
                'ADC_Canal_R': round(f['intensidade_roi'], 1),
                'ADC_Total_Canal_R': round(f['intensidade_total'], 1),
                'Centro_X': f['centro'][0], 'Centro_Y': f['centro'][1]
            } for f in todos_filmes])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV Completo", csv, "filmes_resultado.csv", "text/csv")
            
            # ==================== CURVA DE CALIBRACAO ====================
            st.markdown("---")
            st.header("📊 Curva de Calibração")
            
            st.error("🚨 **OBRIGATORIO: Voce DEVE incluir um filme com DOSE = 0 Gy (filme NAO IRRADIADO).** Sem o filme de referencia, o NOD nao pode ser calculado corretamente.")
            st.error("🚨 **IMPORTANTE: Use apenas arquivos TIF originais do scanner (16-bit / 48-bit color).** Screenshots, JPGs ou PNGs exportados perdem a precisão e geram resultados absurdos (ADC na faixa de 0-255 em vez de 10.000-60.000).")
            
            st.info(f"Tipo de filme selecionado: **{tipo_filme}**")
            st.info(f"Total de filmes disponíveis: {len(todos_filmes)}")
            st.warning("⚠️ Selecione pelo menos 3 filmes de calibração com doses conhecidas, incluindo UM filme de 0 Gy.")
            
            # Selecionar filmes para calibração
            st.subheader("Selecione os filmes de calibração e informe as doses")
            
            # Modo manual de ADC
            usar_adc_manual = st.checkbox("🔧 Usar valores de ADC manualmente (ignorar leitura do scan)", value=False,
                                          help="Se os valores lidos do TIFF parecem errados, marque esta opção e digite os ADCs manualmente para cada filme.")
            if usar_adc_manual:
                st.info("Modo manual ativado. Digite os valores de ADC que você obteve do scanner/software de referência.")
            
            filmes_calibracao = []
            
            for i, filme in enumerate(todos_filmes):
                if usar_adc_manual:
                    col_check, col_info, col_adc, col_dose = st.columns([1, 2, 2, 2])
                else:
                    col_check, col_info, col_dose = st.columns([1, 3, 2])
                    
                with col_check:
                    usar = st.checkbox(f"Usar", key=f"calib_{i}")
                with col_info:
                    st.markdown(f"**Filme {filme['id']}** | ROI: {filme['roi_cm']:.1f} cm")
                    if not usar_adc_manual:
                        st.caption(f"ADC auto: {filme['intensidade_roi']:.1f}")
                
                if usar_adc_manual:
                    with col_adc:
                        adc_manual = st.number_input(f"ADC", min_value=0.0, value=float(filme['intensidade_roi']), step=1.0, key=f"adc_manual_{i}")
                
                with col_dose:
                    dose_val = st.number_input(f"Dose (Gy/cGy)", min_value=0.0, value=0.0, step=0.1, key=f"dose_{i}")
                
                if usar:
                    filme_calib = {
                        'filme': dict(filme),  # copia para nao alterar original
                        'dose': dose_val,
                        'id': filme['id']
                    }
                    if usar_adc_manual:
                        filme_calib['filme']['intensidade_roi'] = adc_manual
                    filmes_calibracao.append(filme_calib)
            
            # Unidade
            unidade = st.radio("Unidade da dose", ["Gy", "cGy"], horizontal=True)
            
            # Tipo de fitting
            tipo_fitting = st.radio("Tipo de fitting", ["Polinomial 2o grau", "Polinomial n (Dosepy)", "Racional (Dosepy/cobaltCorsair)", "Potencia"], horizontal=True, key="fitting_type", 
                                     help="Polinomial 2o: Dose=a*NOD²+b*NOD+c | Polinomial n: Dose=a*NOD+b*NOD^n (Dosepy) | Racional: Dose=-c+b/(NOD-a) (Dosepy/cobaltCorsair) | Potencia: Dose=K1*NOD^K2")
            
            # Botao gerar curva
            if st.button("🔬 GERAR CURVA DE CALIBRAÇÃO", type="primary"):
                if len(filmes_calibracao) < 3:
                    st.error(f"❌ Selecione pelo menos 3 filmes de calibração. Atual: {len(filmes_calibracao)}")
                    st.stop()
                
                with st.spinner("Calculando curva de calibração..."):
                    # Converter dose para Gy se necessario
                    if unidade == "cGy":
                        for f in filmes_calibracao:
                            f['dose'] = f['dose'] / 100.0
                    
                    # ========== VALIDACOES DE SANIDADE ==========
                    adcs_raw = [f['filme']['intensidade_roi'] for f in filmes_calibracao]
                    max_adc = max(adcs_raw)
                    min_adc = min(adcs_raw)
                    
                    # Alerta: imagem parece ser 8-bit ou muito baixa
                    if max_adc < 300:
                        st.error(f"🚨 **VALORES DE ADC MUITO BAIXOS (max={max_adc:.1f})**")
                        st.error("Os scans parecem estar em 8-bit (0-255). Filmes EBT3/EBT4 escaneados profissionalmente devem ter ADC na faixa de 10.000–60.000 (16-bit).")
                        st.info("**Possíveis causas:**\n1. Você carregou screenshots/prints em vez dos arquivos TIF originais do scanner\n2. As imagens foram salvas como JPG/PNG de 8-bit e perderam a precisão do scanner\n3. O scanner não está configurado para 16-bit / 48-bit color")
                        st.warning("⚠️ A curva será gerada, mas NÃO será confiável para dosimetria clínica.")
                    elif max_adc < 1000:
                        st.info(f"ℹ️ ADC na faixa de {min_adc:.0f}–{max_adc:.0f}. Se seu scanner é de transmissão 8-bit ou 12-bit, estes valores são normais. O NOD é o que importa para a calibração.")
                    
                    # Alerta: filme 0 Gy com ADC = 0
                    doses_list = [f['dose'] for f in filmes_calibracao]
                    if 0.0 in doses_list:
                        idx_0 = doses_list.index(0.0)
                        adc_0 = adcs_raw[idx_0]
                        if adc_0 < 1:
                            st.error(f"🚨 O filme de 0 Gy tem ADC = {adc_0:.1f} (quase zero). Isso indica que a imagem está escura demais ou é um print, não o scan original.")
                    
                    # Calcular NOD para TODOS os filmes
                    pv0, nod_info = calcular_nod(filmes_calibracao)
                    
                    # VERIFICACAO CRITICA: existe filme de 0 Gy?
                    if nod_info.get('erro', False):
                        st.error("🚨 " + nod_info['erro_msg'])
                        st.error("⚠️ O filme de 0 Gy (nao irradiado) eh transparente e o algoritmo automatico pode nao detecta-lo.")
                        st.info("💡 SOLUCAO: Use a secao 'Adicionar Filme Nao Detectado Manualmente' acima para adicionar o filme de 0 Gy. Digite as coordenadas (x, y, largura, altura) do filme transparente na imagem.")
                        st.stop()
                    
                    nods = np.array([f['nod'] for f in filmes_calibracao])
                    doses = np.array([f['dose'] for f in filmes_calibracao])
                    adcs = np.array([f['filme']['intensidade_roi'] for f in filmes_calibracao])
                    
                    # Info do calculo
                    st.info(f"**Referência 0 Gy:** ADC = {pv0:.1f} | Dose = {nod_info['dose_0']:.2f} Gy")
                    st.info(f"**Maior dose:** ADC = {nod_info['pv_max']:.1f} | Dose = {nod_info['dose_max']:.2f} Gy")
                    
                    if nod_info['adc_aumenta_com_dose']:
                        st.success("✅ Scanner detectado: ADC **aumenta** com dose (tipo IBA/refletância). NOD = log10(PV/PV₀)")
                    else:
                        st.success("✅ Scanner detectado: ADC **diminui** com dose (transmissão padrão). NOD = log10(PV₀/PV)")
                    
                    # Mostrar tabela de NODs para debug
                    debug_nods = pd.DataFrame([{
                        'Filme': f['id'],
                        'Dose_Gy': f['dose'],
                        'ADC': f['filme']['intensidade_roi'],
                        'NOD': f['nod'],
                        'Forma': f.get('nod_info', '')
                    } for f in filmes_calibracao])
                    st.subheader("Debug — Cálculo de NOD")
                    st.dataframe(debug_nods, use_container_width=True, hide_index=True)
                    
                    # Verificar se NODs são monotonicos crescentes
                    for i in range(1, len(filmes_calibracao)):
                        if nods[i] < nods[i-1] and doses[i] > doses[i-1]:
                            st.warning(f"⚠️ NOD do filme {filmes_calibracao[i]['id']} ({nods[i]:.4f}) < filme anterior ({nods[i-1]:.4f}) apesar de dose maior. Verifique os scans.")
                    
                    # Escolher fitting (ja definido antes do botao)
                    if tipo_fitting == "Polinomial 2o grau":
                        curva = fitting_polinomial2(nods, doses)
                    elif tipo_fitting == "Polinomial n (Dosepy)":
                        curva = fitting_polinomial_n(nods, doses)
                    elif tipo_fitting == "Racional (Dosepy/cobaltCorsair)":
                        curva = fitting_racional(nods, doses)
                    else:
                        curva = fitting_potencia(nods, doses)
                    
                    if curva:
                        # Calcular doses preditas
                        doses_pred = np.array([_calcular_dose_curva(n, curva) for n in nods])
                        
                        # Calcular erros com protecao para divisao por zero
                        erros_gy = doses_pred - doses
                        erros_pct = []
                        for i in range(len(doses)):
                            if doses[i] > 0.001:
                                erros_pct.append(float(abs((doses_pred[i] - doses[i]) / doses[i] * 100)))
                            else:
                                erros_pct.append(0.0)  # Dose ~0 Gy = erro 0%
                        
                        # Criar DataFrame com dados limpos
                        dados_tabela = []
                        for i in range(len(filmes_calibracao)):
                            dados_tabela.append({
                                'Filme': int(filmes_calibracao[i]['id']),
                                'NOD': float(nods[i]),
                                'ADC': float(adcs[i]),
                                'Dose_Real_Gy': float(doses[i]),
                                'Dose_Predita_Gy': float(doses_pred[i]),
                                'Desvio_Gy': float(erros_gy[i]),
                                'Erro_%': float(erros_pct[i])
                            })
                        
                        df_erros = pd.DataFrame(dados_tabela)
                        
                        # Gerar graficos
                        fig_buf_nod = gerar_grafico_nod_dose(filmes_calibracao, curva, tipo_filme)
                        fig_buf_adc = gerar_grafico_adc_dose(filmes_calibracao, tipo_filme)
                        
                        # Salvar na sessao
                        curva_data = {
                            'tipo_filme': tipo_filme,
                            'equacao': curva['equation'],
                            'r2': float(curva['r2']),
                            'dpi': dpi_m,
                            'unidade': 'Gy',
                            'pv0_referencia': float(pv0),
                            'doses_calibracao': doses.tolist(),
                            'nods_calibracao': nods.tolist(),
                            'adcs_calibracao': adcs.tolist()
                        }
                        
                        if 'a' in curva:
                            curva_data['a'] = float(curva['a'])
                            curva_data['b'] = float(curva['b'])
                            curva_data['c'] = float(curva['c'])
                        else:
                            curva_data['K1'] = float(curva['K1'])
                            curva_data['K2'] = float(curva['K2'])
                        
                        st.session_state['resultado_curva'] = {
                            'tipo': tipo_filme,
                            'equation': curva['equation'],
                            'r2': float(curva['r2']),
                            'fig_buf_nod': fig_buf_nod,
                            'fig_buf_adc': fig_buf_adc,
                            'df_erros': df_erros,
                            'curva_data': curva_data
                        }
                
                st.rerun()
            
            # Mostrar resultados da curva (FORA do botao, usando session_state)
            if 'resultado_curva' in st.session_state:
                resultado = st.session_state['resultado_curva']
                
                st.success("✅ Curva de calibração gerada com sucesso!")
                
                # Mostrar dois graficos lado a lado
                col_graf1, col_graf2 = st.columns(2)
                with col_graf1:
                    st.subheader("📈 NOD vs Dose")
                    st.image(resultado['fig_buf_nod'], use_container_width=True)
                with col_graf2:
                    st.subheader("📉 ADC vs Dose")
                    st.image(resultado['fig_buf_adc'], use_container_width=True)
                
                # Mostrar equacao
                st.info(f"**Equação:** {resultado['equation']}")
                st.info(f"**R²:** {resultado['r2']:.6f}")
                
                # Tabela de erros
                st.subheader("Tabela de Erros")
                df_erros = resultado['df_erros'].fillna(0.0)
                st.dataframe(df_erros, use_container_width=True, hide_index=True)
                
                # Salvar curva na sessao permanente
                st.session_state['curva_calibracao'] = resultado['curva_data']
                
                # Download da curva
                curva_json = json.dumps(resultado['curva_data'], indent=2)
                st.download_button(
                    "Download Curva de Calibracao (.json)",
                    curva_json,
                    f"curva_calibracao_{tipo_filme}.json",
                    "application/json"
                )
