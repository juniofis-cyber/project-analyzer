"""
Project Analyzer - Multi-Otsu para detectar regiões irradiadas
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

from skimage.color import rgb2gray
from skimage.filters import threshold_multiotsu
from skimage.morphology import remove_small_objects, closing, square
from skimage.measure import label, regionprops

st.set_page_config(page_title="Project Analyzer", page_icon="🔬", layout="wide")

st.title("🔬 Project Analyzer")

def detectar_regioes(imagem, area_min):
    """Detecta regiões irradiadas usando Multi-Otsu"""
    gray = rgb2gray(imagem)
    
    # Multi-Otsu com 3 classes: fundo | filme | irradiado
    thresholds = threshold_multiotsu(gray, classes=3)
    
    # Pegar apenas as regiões irradiadas (mais escuras que thresholds[1])
    binary = gray < thresholds[1]
    
    # Limpeza
    binary = remove_small_objects(binary, min_size=int(area_min))
    binary = closing(binary, square(5))
    
    # Labeling
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray)
    
    regioes = []
    for r in regions:
        if r.area >= area_min:
            minr, minc, maxr, maxc = r.bbox
            regioes.append({
                'area': r.area,
                'intensidade_media': r.mean_intensity,
                'centro': (int(r.centroid[1]), int(r.centroid[0])),
                'bbox': (minc, minr, maxc - minc, maxr - minr)
            })
    
    return regioes, thresholds

def ordenar(regioes):
    ordenadas = sorted(regioes, key=lambda x: x['intensidade_media'], reverse=True)
    for i, r in enumerate(ordenadas, 1):
        r['id_ordenado'] = i
    return ordenadas

def desenhar(imagem, regioes):
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
    
    for r in regioes:
        x, y, w, h = r['bbox']
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
        cx, cy = r['centro']
        bbox = draw.textbbox((0, 0), str(r['id_ordenado']), font=fonte)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([cx - tw//2 - 4, cy - th//2 - 4, cx + tw//2 + 4, cy + th//2 + 4], fill=(0, 0, 0))
        draw.text((cx - tw//2, cy - th//2), str(r['id_ordenado']), fill=(255, 255, 255), font=fonte)
    
    return np.array(img_pil)

# Interface
with st.sidebar:
    st.header("Configurações")
    dpi = st.number_input("DPI", 1, 2400, 50)
    area_min = st.slider("Área Mínima (pixels)", 100, 50000, 10000, 500)

st.header("Upload da Imagem")
arquivo = st.file_uploader("Imagem EBT3", type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])

if arquivo:
    img = Image.open(io.BytesIO(arquivo.read()))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_orig = np.array(img)
    img_norm = img_orig / 255.0 if img_orig.max() > 1 else img_orig
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(img_orig, use_column_width=True)
    
    if st.button("ANALISAR", type="primary"):
        with st.spinner("Processando..."):
            regioes, thresh = detectar_regioes(img_norm, area_min)
            
            st.info(f"Thresholds: {thresh[0]:.3f}, {thresh[1]:.3f}")
            
            if not regioes:
                st.warning("Nenhuma região! Reduza Área Mínima")
                st.stop()
            
            reg_ord = ordenar(regioes)
            img_res = desenhar(img_norm, reg_ord)
            
            with col2:
                st.subheader(f"{len(regioes)} regiões")
                st.image(img_res, use_column_width=True)
            
            # Tabela
            mm = 25.4 / dpi
            df = pd.DataFrame([{
                'N': r['id_ordenado'],
                'Area_mm2': round(r['area'] * mm**2, 2),
                'Intensidade': round(r['intensidade_media'], 4)
            } for r in reg_ord])
            
            st.dataframe(df, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "resultado.csv", "text/csv")
else:
    st.info("Faça upload de uma imagem")
