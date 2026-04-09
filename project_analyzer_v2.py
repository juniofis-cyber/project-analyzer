"""
Project Analyzer v5.0
Detecção de múltiplos níveis de radiação usando Multi-Otsu
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.morphology import erosion, dilation, remove_small_objects, remove_small_holes
from skimage.morphology import square, disk
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

st.set_page_config(page_title="Project Analyzer", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .subtitle { font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

def detectar_fundo(imagem):
    """Detecta e corta o filme da imagem"""
    gray = rgb2gray(imagem)
    thresh = threshold_otsu(gray)
    binary = gray < thresh
    binary = clear_border(binary)
    binary = erosion(binary, square(5))
    binary = dilation(binary, square(3))
    binary = remove_small_objects(binary, min_size=1000)
    binary = remove_small_holes(binary, area_threshold=1000)
    labeled = label(binary)
    regions = regionprops(labeled)
    if not regions:
        return imagem, None
    largest = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox
    margem = 20
    h, w = imagem.shape[:2]
    minr = max(0, minr - margem)
    minc = max(0, minc - margem)
    maxr = min(h, maxr + margem)
    maxc = min(w, maxc + margem)
    return imagem[minr:maxr, minc:maxc], (minr, minc, maxr, maxc)

def detectar_regioes_multiotsu(imagem, area_min_percent=0.5, n_classes=4, indice_classe=1):
    """
    Detecta regiões irradiadas usando Multi-Otsu
    
    n_classes: número de níveis de intensidade (4 = fundo, claro, médio, escuro)
    indice_classe: qual(is) classe(s) considerar como irradiada
                   1 = só mais escuras
                   2 = médias e escuras  
                   3 = clara, média e escura (todas menos fundo)
    """
    gray = rgb2gray(imagem)
    area_total = imagem.shape[0] * imagem.shape[1]
    area_minima = (area_min_percent / 100) * area_total
    
    try:
        # Multi-Otsu: separa em n_classes níveis
        thresholds = threshold_multiotsu(gray, classes=n_classes)
        
        # thresholds[0] = separa fundo do resto
        # thresholds[1] = separa claro do médio  
        # thresholds[2] = separa médio do escuro (se n_classes=4)
        
        # Pegar regiões baseado no índice
        if indice_classe == 1:
            # Só a classe mais escura
            binary = gray < thresholds[-1]
        elif indice_classe == 2:
            # Médio + escuro
            binary = gray < thresholds[-2]
        else:
            # Todas as classes exceto fundo (claro + médio + escuro)
            binary = gray < thresholds[0]
            
    except:
        # Fallback para Otsu simples
        thresh = threshold_otsu(gray)
        binary = gray < thresh
    
    # Limpeza
    binary = remove_small_objects(binary, min_size=int(area_minima))
    binary = remove_small_holes(binary, area_threshold=int(area_minima/2))
    binary = erosion(binary, disk(3))
    binary = dilation(binary, disk(2))
    
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray)
    
    regioes = []
    for region in regions:
        if region.area < area_minima:
            continue
        regioes.append({
            'area': region.area,
            'intensidade_media': region.mean_intensity,
            'centro': (int(region.centroid[1]), int(region.centroid[0])),
            'bbox': region.bbox,
        })
    
    return regioes

def ordenar_por_escurecimento(regioes):
    regioes_ordenadas = sorted(regioes, key=lambda x: x['intensidade_media'], reverse=True)
    for i, regiao in enumerate(regioes_ordenadas, 1):
        regiao['id_ordenado'] = i
    return regioes_ordenadas

def desenhar_resultado(imagem, regioes):
    if imagem.max() <= 1:
        img_uint8 = (imagem * 255).astype(np.uint8)
    else:
        img_uint8 = imagem.astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(img_pil)
    try:
        fonte = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
    except:
        fonte = ImageFont.load_default()
    for regiao in regioes:
        minr, minc, maxr, maxc = regiao['bbox']
        draw.rectangle([minc, minr, maxc, maxr], outline=(0, 255, 0), width=3)
        cx, cy = regiao['centro']
        numero = str(regiao['id_ordenado'])
        bbox = draw.textbbox((0, 0), numero, font=fonte)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([cx-tw//2-4, cy-th//2-4, cx+tw//2+4, cy+th//2+4], fill=(0, 0, 0))
        draw.text((cx-tw//2, cy-th//2), numero, fill=(255, 255, 255), font=fonte)
    return np.array(img_pil)

def criar_dataframe(regioes, dpi=50):
    mm_por_pixel = 25.4 / dpi
    dados = []
    for r in regioes:
        area_mm2 = r['area'] * (mm_por_pixel ** 2)
        dados.append({
            'Número': r['id_ordenado'],
            'Área (mm²)': round(area_mm2, 2),
            'Área (pixels)': int(r['area']),
            'Intensidade Média': round(r['intensidade_media'], 4),
            'Centro X': r['centro'][0],
            'Centro Y': r['centro'][1]
        })
    return pd.DataFrame(dados)

st.markdown('<p class="main-title">🔬 Project Analyzer v5.0</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multi-Otsu para múltiplos níveis de radiação</p>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configurações")
    dpi = st.number_input("DPI do Scanner", 1, 2400, 50)
    area_min = st.slider("Área Mínima (%)", 0.01, 5.0, 0.5, 0.01, help="Regiões menores são ignoradas")
    
    n_classes = st.selectbox(
        "Número de Classes",
        [3, 4, 5],
        index=1,
        help="4 = fundo, claro, médio, escuro"
    )
    
    indice_classe = st.selectbox(
        "Classes a Detectar",
        [1, 2, 3],
        index=2,
        format_func=lambda x: {
            1: "1 - Só escuras",
            2: "2 - Médias + escuras", 
            3: "3 - Todas (claro+médio+escuro)"
        }[x],
        help="Selecione quais níveis de radiação detectar"
    )
    
    st.markdown("---")
    st.info("💡 Dica: Use 'Todas' para detectar as 4 regiões")

st.header("📁 Upload da Imagem")
arquivo = st.file_uploader("Selecione a imagem do filme EBT3", type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])

if arquivo:
    img_pil = Image.open(io.BytesIO(arquivo.read()))
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    img_original = np.array(img_pil)
    if img_original.max() > 1:
        img_normalized = img_original / 255.0
    else:
        img_normalized = img_original
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagem Original")
        st.image(img_original, use_column_width=True)
    
    if st.button("🔍 ANALISAR", type="primary", use_container_width=True):
        with st.spinner("Processando com Multi-Otsu..."):
            img_filme, bbox = detectar_fundo(img_normalized)
            if bbox is None:
                st.error("❌ Não foi possível detectar o filme!")
                st.stop()
            st.info(f"✅ Filme detectado: {img_filme.shape[1]}x{img_filme.shape[0]} px")
            
            regioes = detectar_regioes_multiotsu(img_filme, area_min, n_classes, indice_classe)
            
            if not regioes:
                st.warning("⚠️ Nenhuma região detectada!")
                st.info("Tente mudar 'Classes a Detectar' para 'Todas'")
                st.stop()
            
            regioes_ord = ordenar_por_escurecimento(regioes)
            img_resultado = desenhar_resultado(img_filme, regioes_ord)
            df = criar_dataframe(regioes_ord, dpi)
            
            with col2:
                st.subheader(f"Resultado: {len(regioes)} regiões")
                st.image(img_resultado, use_column_width=True)
            
            st.markdown("---")
            st.header("📊 Análise")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Regiões", len(regioes))
            c2.metric("DPI", dpi)
            c3.metric("Resolução", f"{25.4/dpi:.2f} mm/px")
            c4.metric("Ordenação", "1=Claro → N=Escuro")
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, f"resultado_{arquivo.name.split('.')[0]}.csv", "text/csv", use_container_width=True)
            with col_d2:
                buf = io.BytesIO()
                Image.fromarray(img_resultado).save(buf, format='PNG')
                st.download_button("📥 Download Imagem", buf.getvalue(), f"analisado_{arquivo.name.split('.')[0]}.png", "image/png", use_container_width=True)
else:
    st.info("👆 Faça upload de uma imagem para começar!")
