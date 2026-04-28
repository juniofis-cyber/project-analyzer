import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
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
    """Desenha contornos na imagem ORIGINAL do scanner"""
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
        # Contorno do filme (VERDE)
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
    """Desenha contorno verde, recuo vermelho tracejado e ROI azul na imagem INDIVIDUAL do filme"""
    if imagem_filme.max() <= 1:
        img = (imagem_filme * 255).astype(np.uint8)
    else:
        img = imagem_filme.astype(np.uint8)
    
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    h, w = img.shape[:2]
    
    # 1. CONTORNO DO FILME (VERDE)
    draw.rectangle([0, 0, w-1, h-1], outline=(0, 255, 0), width=2)
    
    # 2. Recuo 5mm (VERMELHO TRACEJADO FINO)
    if recuo_px > 0:
        x_recuo = recuo_px
        y_recuo = recuo_px
        w_recuo = w - 2 * recuo_px
        h_recuo = h - 2 * recuo_px
        if w_recuo > 0 and h_recuo > 0:
            desenhar_tracejado_fino(draw, x_recuo, y_recuo, x_recuo + w_recuo, y_recuo + h_recuo, (255, 0, 0), 1)
    
    # 3. ROI (AZUL)
    if roi_bbox:
        rx, ry, rw, rh = roi_bbox
        draw.rectangle([rx, ry, rx + rw, ry + rh], outline=(0, 102, 255), width=2)
    
    return np.array(img_pil)

def desenhar_tracejado_fino(draw, x1, y1, x2, y2, cor, largura, segmento=5, espaco=3):
    """Desenha retangulo tracejado com segmentos menores e mais finos"""
    # Topo
    i = x1
    while i < x2:
        seg = min(segmento, x2 - i)
        draw.line([(i, y1), (i + seg, y1)], fill=cor, width=largura)
        i += segmento + espaco
    
    # Base
    i = x1
    while i < x2:
        seg = min(segmento, x2 - i)
        draw.line([(i, y2), (i + seg, y2)], fill=cor, width=largura)
        i += segmento + espaco
    
    # Esquerda
    i = y1
    while i < y2:
        seg = min(segmento, y2 - i)
        draw.line([(x1, i), (x1, i + seg)], fill=cor, width=largura)
        i += segmento + espaco
    
    # Direita
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

st.title("🔬 Project Analyzer v7.1")
st.markdown("Recuo AAPM TG-55 (5mm) | ROI proporcional (60% da menor dimensao) | Tracejado fino")

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
            st.dataframe(df, use_container_width=True, hide_index=True)
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
                col_orig, col_masc = st.columns([4, 1])
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
                        
                        # Desenhar recuo + ROI no filme individual
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
