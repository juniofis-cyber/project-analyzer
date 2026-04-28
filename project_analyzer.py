import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
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
    gray = rgb2gray(imagem)
    thresh = threshold_otsu(gray)
    binary = gray < thresh
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
    gray = rgb2gray(imagem)
    thresh = threshold_otsu(gray)
    thresh_ajustado = thresh * (1 - offset)
    binary = gray < thresh_ajustado
    if erosao > 0:
        binary = erosion(binary, square(erosao))
    binary = remove_small_objects(binary, min_size=area_min)
    if fechamento > 0:
        binary = closing(binary, square(fechamento))
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray)
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
    return regioes, gray

def detectar_filmes_multiplos(imagem, area_min):
    gray = rgb2gray(imagem)
    thresh = threshold_otsu(gray)
    binary = gray < thresh
    binary = clear_border(binary)
    binary = remove_small_objects(binary, min_size=area_min)
    binary = closing(binary, square(5))
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray)
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
    if len(imagem_filme.shape) == 3:
        gray = rgb2gray(imagem_filme)
    else:
        gray = imagem_filme
    roi_pixels = gray[y1:y2, x1:x2]
    intensidade_roi = np.mean(roi_pixels)
    intensidade_total = np.mean(gray)
    bbox_roi = (x1, y1, x2 - x1, y2 - y1)
    return intensidade_roi, bbox_roi, intensidade_total

def desenhar_marcacoes_original(imagem, filmes, dpi, mostrar_recuo=True, mostrar_roi=True):
    if imagem.max() <= 1:
        img = (imagem * 255).astype(np.uint8)
    else:
        img = imagem.astype(np.uint8)
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
    if imagem_filme.max() <= 1:
        img = (imagem_filme * 255).astype(np.uint8)
    else:
        img = imagem_filme.astype(np.uint8)
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
    """Calcula NOD: NOD = log10(PV0 / PVirradiado)"""
    filme_0 = max(filmes_calibracao, key=lambda f: f['filme']['intensidade_roi'])
    pv0 = filme_0['filme']['intensidade_roi']
    
    for f in filmes_calibracao:
        pv_irrad = f['filme']['intensidade_roi']
        if pv_irrad > 0 and pv0 > pv_irrad:
            f['nod'] = np.log10(pv0 / pv_irrad)
        else:
            f['nod'] = 0.0
    
    return pv0

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

def gerar_grafico_nod_dose(filmes, curva, tipo_filme):
    """Grafico NOD vs Dose (padrao cientifico)"""
    nods = np.array([f['nod'] for f in filmes])
    doses = np.array([f['dose'] for f in filmes])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(nods, doses, color='red', s=150, label='Dados medidos', zorder=5, edgecolors='black', linewidth=1)
    
    nods_fit = np.linspace(0, max(nods)*1.15, 200)
    if 'a' in curva:
        doses_fit = curva['a'] * nods_fit**2 + curva['b'] * nods_fit + curva['c']
    else:
        doses_fit = curva['K1'] * (nods_fit ** curva['K2'])
    
    ax.plot(nods_fit, doses_fit, 'b-', linewidth=2.5, label='Curva ajustada', zorder=3)
    
    for i in range(len(nods)):
        if 'a' in curva:
            dose_na_curva = curva['a'] * nods[i]**2 + curva['b'] * nods[i] + curva['c']
        else:
            dose_na_curva = curva['K1'] * (nods[i] ** curva['K2'])
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
    """Grafico ADC (Pixel Value) vs Dose - relacao direta"""
    adcs = np.array([f['filme']['intensidade_roi'] for f in filmes])
    doses = np.array([f['dose'] for f in filmes])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(adcs, doses, color='darkorange', s=150, label='Dados medidos', zorder=5, edgecolors='black', linewidth=1)
    
    # Fitting polinomial 2a ordem para ADC vs Dose
    try:
        coefs = np.polyfit(adcs, doses, 2)
        a, b, c = coefs
        adcs_fit = np.linspace(min(adcs)*0.9, max(adcs)*1.05, 200)
        doses_fit = a * adcs_fit**2 + b * adcs_fit + c
        ax.plot(adcs_fit, doses_fit, 'purple', linewidth=2.5, label=f'Dose = {a:.4f}*ADC² + {b:.4f}*ADC + {c:.4f}', zorder=3)
        
        # Calcular R2
        doses_pred = a * adcs**2 + b * adcs + c
        ss_res = np.sum((doses - doses_pred)**2)
        ss_tot = np.sum((doses - np.mean(doses))**2)
        r2_adc = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    except:
        r2_adc = 0
    
    ax.set_xlabel('ADC (Average Digitized Count)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dose (Gy)', fontsize=14, fontweight='bold')
    ax.set_title(f'Curva de Calibração {tipo_filme} - ADC vs Dose\nR² = {r2_adc:.6f}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_ylim(bottom=0)
    
    # Inverter eixo X (ADC diminui com dose)
    ax.invert_xaxis()
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ==================== INTERFACE ====================

st.title("🔬 Project Analyzer v8.2")
st.markdown("NOD (Net Optical Density) | Curva crescente | ADC vs Dose")

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
        img = Image.open(io.BytesIO(arquivo.read()))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_orig = np.array(img)
        img_norm = img_orig/255.0 if img_orig.max() > 1 else img_orig
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img_orig, use_column_width=True)
        
        if st.button("ANALISAR", type="primary", key="btn_unico"):
            with st.spinner("Processando..."):
                img_filme = cortar_filme_unico(img_norm)
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
                st.markdown(f"**Regiao {i+1}** (Intensidade: {regiao['intensidade']:.4f})")
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
            img_res = desenhar_marcacoes_original(img_filme, reg_ord, dpi_s)
            st.subheader("Resultado Final")
            st.image(img_res, use_column_width=True)
            cm = 1 / (dpi_s / 2.54)
            df = pd.DataFrame([{
                'Filme': r['id'], 'Area_cm2': round(r['area'] * cm * cm, 2),
                'Largura_cm': round(r['bbox'][2] * cm, 2),
                'Altura_cm': round(r['bbox'][3] * cm, 2),
                'Intensidade': round(r['intensidade'], 4), 'Razao': round(r['razao'], 2)
            } for r in reg_ord])
            st.dataframe(df, use_column_width=True, hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False), "resultado.csv", "text/csv")

# ==================== MODO VARIOS FILMES ====================

else:
    st.header("Upload dos Filmes Escaneados")
    st.info("Dica: Segure Ctrl e clique para selecionar multiplos arquivos")
    arquivos = st.file_uploader("Envie uma ou mais imagens com os filmes",
                               type=['tif','tiff','png','jpg','jpeg'],
                               accept_multiple_files=True)
    
    if arquivos and len(arquivos) > 0:
        st.success(f"{len(arquivos)} arquivo(s) carregado(s)")
        
        if st.button("DETECTAR FILMES", type="primary"):
            with st.spinner("Detectando filmes..."):
                todos_filmes = []
                imagens_originais = []
                
                for idx_arq, arquivo in enumerate(arquivos):
                    img = Image.open(io.BytesIO(arquivo.read()))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_np = np.array(img)
                    img_norm = img_np/255.0 if img_np.max() > 1 else img_np
                    
                    filmes, binary = detectar_filmes_multiplos(img_norm, area_min_multi)
                    
                    for f in filmes:
                        f['arquivo'] = arquivo.name
                        roi_px, roi_cm = calcular_roi_quadrado(f['bbox'][2], f['bbox'][3], dpi)
                        f['roi_px'] = roi_px
                        f['roi_cm'] = roi_cm
                        intens_roi, bbox_roi, intens_total = calcular_intensidade_roi(f['imagem'], roi_px)
                        f['intensidade_roi'] = intens_roi
                        f['intensidade_total'] = intens_total
                        f['roi_bbox'] = bbox_roi
                    
                    filmes_temp = ordenar(filmes)
                    img_com_contornos = desenhar_marcacoes_original(
                        img_norm, filmes_temp, dpi, mostrar_recuo, mostrar_roi
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
                    st.image(img_info['imagem'], use_column_width=True)
                with col_masc:
                    st.markdown("**Mascara de deteccao**")
                    st.image((img_info['mascara'] * 255).astype(np.uint8), use_column_width=True)
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
                            st.image(img_com_marcas, use_column_width=True)
                            st.caption(f"ROI: {filme['roi_cm']:.1f} cm | Int ROI: {filme['intensidade_roi']:.4f}")
            
            # Tabela completa
            st.markdown("---")
            st.header("Tabela Completa")
            
            df = pd.DataFrame([{
                'Filme': f['id'], 'Arquivo': f['arquivo'],
                'Area_cm2': round(f['area'] / (px_por_cm ** 2), 2),
                'Largura_cm': round(f['bbox'][2] / px_por_cm, 2),
                'Altura_cm': round(f['bbox'][3] / px_por_cm, 2),
                'ROI_cm': round(f['roi_cm'], 2),
                'Intensidade_ROI': round(f['intensidade_roi'], 4),
                'Intensidade_Total': round(f['intensidade_total'], 4),
                'Centro_X': f['centro'][0], 'Centro_Y': f['centro'][1]
            } for f in todos_filmes])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV Completo", csv, "filmes_resultado.csv", "text/csv")
            
            # ==================== CURVA DE CALIBRACAO ====================
            st.markdown("---")
            st.header("📊 Curva de Calibração")
            
            st.info(f"Tipo de filme selecionado: **{tipo_filme}**")
            st.info(f"Total de filmes disponíveis: {len(todos_filmes)}")
            st.warning("⚠️ Selecione pelo menos 3 filmes de calibração com doses conhecidas")
            
            # Selecionar filmes para calibração
            st.subheader("Selecione os filmes de calibração e informe as doses")
            
            filmes_calibracao = []
            
            for i, filme in enumerate(todos_filmes):
                col_check, col_info, col_dose = st.columns([1, 3, 2])
                with col_check:
                    usar = st.checkbox(f"Usar", key=f"calib_{i}")
                with col_info:
                    st.markdown(f"**Filme {filme['id']}** | Int ROI: {filme['intensidade_roi']:.4f} | ROI: {filme['roi_cm']:.1f} cm")
                with col_dose:
                    dose_val = st.number_input(f"Dose (Gy/cGy)", min_value=0.0, value=0.0, step=0.1, key=f"dose_{i}")
                
                if usar:
                    filmes_calibracao.append({
                        'filme': filme,
                        'dose': dose_val,
                        'id': filme['id']
                    })
            
            # Unidade
            unidade = st.radio("Unidade da dose", ["Gy", "cGy"], horizontal=True)
            
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
                    
                    # Calcular NOD para TODOS os filmes
                    pv0 = calcular_nod(filmes_calibracao)
                    nods = np.array([f['nod'] for f in filmes_calibracao])
                    doses = np.array([f['dose'] for f in filmes_calibracao])
                    adcs = np.array([f['filme']['intensidade_roi'] for f in filmes_calibracao])
                    
                    st.info(f"PV do filme 0 Gy (referência): {pv0:.4f}")
                    
                    # Escolher fitting conforme tipo de filme
                    if tipo_filme == "EBT3":
                        curva = fitting_polinomial2(nods, doses)
                    else:  # EBT4
                        curva = fitting_potencia(nods, doses)
                    
                    if curva:
                        # Calcular doses preditas
                        if 'a' in curva:
                            doses_pred = curva['a'] * nods**2 + curva['b'] * nods + curva['c']
                        else:
                            doses_pred = curva['K1'] * (nods ** curva['K2'])
                        
                        # Calcular erros com protecao para divisao por zero
                        erros_gy = doses_pred - doses
                        erros_pct = []
                        for i in range(len(doses)):
                            if doses[i] > 0:
                                erros_pct.append((doses_pred[i] - doses[i]) / doses[i] * 100)
                            else:
                                erros_pct.append(0.0)  # Dose 0 Gy = erro 0%
                        
                        df_erros = pd.DataFrame({
                            'Filme': [f['id'] for f in filmes_calibracao],
                            'NOD': nods,
                            'ADC': adcs,
                            'Dose_Real_Gy': doses,
                            'Dose_Predita_Gy': doses_pred,
                            'Erro_Gy': erros_gy,
                            'Erro_%': erros_pct
                        })
                        
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
                    st.image(resultado['fig_buf_nod'], use_column_width=True)
                with col_graf2:
                    st.subheader("📉 ADC vs Dose")
                    st.image(resultado['fig_buf_adc'], use_column_width=True)
                
                # Mostrar equacao
                st.info(f"**Equação:** {resultado['equation']}")
                st.info(f"**R²:** {resultado['r2']:.6f}")
                
                # Tabela de erros
                st.subheader("Tabela de Erros")
                st.dataframe(resultado['df_erros'], use_column_width=True, hide_index=True)
                
                # Salvar curva na sessao permanente
                st.session_state['curva_calibracao'] = resultado['curva_data']
                
                # Download da curva
                curva_json = json.dumps(resultado['curva_data'], indent=2)
                st.download_button(
                    "💾 Download Curva de Calibração (.json)",
                    curva_json,
                    f"curva_calibracao_{tipo_filme}.json",
                    "application/json"
                )
