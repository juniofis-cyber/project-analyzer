"""
Project Analyzer - Detecção de regiões irradiadas em EBT3
Baseado em técnicas de OMG Dosimetry e radiochromic-film
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects, closing, square, erosion, dilation
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

st.set_page_config(page_title="Project Analyzer", page_icon="🔬", layout="wide")

st.title("🔬 Project Analyzer")
st.markdown("Detecção de regiões irradiadas em filme EBT3")

def detectar_fundo(imagem):
    """Detecta e corta o filme da imagem"""
    gray = rgb2gray(imagem)
    
    # Otsu para separar fundo branco do filme
    thresh = threshold_otsu(gray)
    binary = gray < thresh
    
    # Limpar bordas
    binary = clear_border(binary)
    
    # Labeling
    labeled = label(binary)
    regions = regionprops(labeled)
    
    if not regions:
        return None
    
    # Pegar o maior contorno (o filme)
    largest = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox
    
    # Cortar com margem
    margem = 10
    h, w = imagem.shape[:2]
    minr = max(0, minr - margem)
    minc = max(0, minc - margem)
    maxr = min(h, maxr + margem)
    maxc = min(w, maxc + margem)
    
    return imagem[minr:maxr, minc:maxc]

def detectar_regioes(imagem, area_min, threshold_offset):
    """
    Detecta regiões irradiadas
    threshold_offset: ajuste do threshold (0-1, maior = mais sensível)
    """
    gray = rgb2gray(imagem)
    
    # Calcular threshold baseado na imagem
    thresh = threshold_otsu(gray)
    
    # Ajustar threshold com offset
    thresh_ajustado = thresh * (1 - threshold_offset)
    
    # Detectar regiões escuras
    binary = gray < thresh_ajustado
    
    # Remover objetos muito pequenos
    binary = remove_small_objects(binary, min_size=area_min)
    
    # Fechar buracos
    binary = closing(binary, square(5))
    
    # Erosão para separar regiões que se tocam
    binary = erosion(binary, square(3))
    
    # Dilatação para recuperar tamanho
    binary = dilation(binary, square(3))
    
    # Labeling
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray)
    
    regioes = []
    for r in regions:
        if r.area >= area_min:
            minr, minc, maxr, maxc = r.bbox
            w = maxc - minc
            h = maxr - minr
            
            # Calcular razão de aspecto (quadrados têm razão próxima de 1)
            razao = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            
            regioes.append({
                'area': r.area,
                'intensidade_media': r.mean_intensity,
                'centro': (int(r.centroid[1]), int(r.centroid[0])),
                'bbox': (minc, minr, w, h),
                'razao': razao
            })
    
    return regioes, thresh, thresh_ajustado

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
    st.header("⚙️ Configurações")
    dpi = st.number_input("DPI", 1, 2400, 50)
    area_min = st.slider("Área Mínima (pixels)", 100, 50000, 3000, 100)
    threshold_offset = st.slider("Sensibilidade", 0.0, 0.5, 0.15, 0.01, 
                                 help="0 = menos sensível | 0.5 = mais sensível")

st.header("📁 Upload da Imagem")
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
    
    if st.button("🔍 ANALISAR", type="primary"):
        with st.spinner("Processando..."):
            # Cortar filme
            img_filme = detectar_fundo(img_norm)
            
            if img_filme is None:
                st.error("❌ Não foi possível detectar o filme!")
                st.stop()
            
            st.info(f"Filme: {img_filme.shape[1]}x{img_filme.shape[0]} px")
            
            # Detectar regiões
            regioes, thresh_original, thresh_ajustado = detectar_regioes(img_filme, area_min, threshold_offset)
            
            st.info(f"Threshold: {thresh_original:.3f} → {thresh_ajustado:.3f}")
            
            if not regioes:
                st.warning("⚠️ Nenhuma região! Aumente Sensibilidade ou reduza Área Mínima")
                st.stop()
            
            # Ordenar e desenhar
            reg_ord = ordenar(regioes)
            img_res = desenhar(img_filme, reg_ord)
            
            with col2:
                st.subheader(f"{len(regioes)} regiões")
                st.image(img_res, use_column_width=True)
            
            # Tabela
            mm = 25.4 / dpi
            df = pd.DataFrame([{
                'N': r['id_ordenado'],
                'Area_mm2': round(r['area'] * mm**2, 2),
                'Intensidade': round(r['intensidade_media'], 4),
                'Razao': round(r['razao'], 2)
            } for r in reg_ord])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "resultado.csv", "text/csv")
else:
    st.info("👆 Faça upload de uma imagem")
area'] * mm**2, 2),
                    'Intensidade': round(r['intensidade_media'], 4)
                } for r in reg_ord])
                
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "resultado.csv", "text/csv")
else:
    st.info("👆 Envie o filme base (não irradiado)")
