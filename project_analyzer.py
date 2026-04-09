"""
Project Analyzer - Calibração com filme base
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects, closing, square
from skimage.measure import label, regionprops

st.set_page_config(page_title="Project Analyzer", page_icon="🔬", layout="wide")

st.title("🔬 Project Analyzer")
st.markdown("Calibração com filme base não irradiado")

def calcular_referencia(imagem):
    """Calcula a intensidade média do filme base (não irradiado)"""
    gray = rgb2gray(imagem)
    return np.mean(gray), np.std(gray)

def detectar_regioes(imagem, media_ref, desvio_ref, fator_desvio, area_min):
    """
    Detecta regiões que são significativamente mais escuras que o filme base
    
    fator_desvio: quantos desvios abaixo da média considerar como irradiado
                  2.0 = 2 desvios (padrão)
                  1.0 = 1 desvio (mais sensível)
                  3.0 = 3 desvios (menos sensível)
    """
    gray = rgb2gray(imagem)
    
    # Limite: média do filme base - fator_desvio * desvio
    limite = media_ref - fator_desvio * desvio_ref
    
    # Detectar pixels abaixo do limite (mais escuros que o filme base)
    binary = gray < limite
    
    # Limpeza morfológica
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
    
    return regioes, limite

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
    dpi = st.number_input("DPI do Scanner", 1, 2400, 50)
    area_min = st.slider("Área Mínima (pixels)", 100, 50000, 5000, 500)
    fator_desvio = st.slider("Sensibilidade (σ)", 0.5, 5.0, 2.0, 0.1, 
                            help="1.0 = mais sensível | 2.0 = padrão | 3.0 = menos sensível")

# PASSO 1: Upload do filme base
st.header("Passo 1: Filme Base (Não Irradiado)")
arquivo_base = st.file_uploader("Envie o filme SEM radiação", 
                                type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
                                key="base")

if arquivo_base:
    img_base = Image.open(io.BytesIO(arquivo_base.read()))
    if img_base.mode != 'RGB':
        img_base = img_base.convert('RGB')
    img_base_np = np.array(img_base)
    img_base_norm = img_base_np / 255.0 if img_base_np.max() > 1 else img_base_np
    
    # Calcular referência
    media_ref, desvio_ref = calcular_referencia(img_base_norm)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Filme Base")
        st.image(img_base_np, use_column_width=True)
    
    with col2:
        st.success(f"✅ Referência calibrada!")
        st.info(f"Média: {media_ref:.4f}\nDesvio: {desvio_ref:.4f}")
    
    # PASSO 2: Upload do filme irradiado
    st.header("Passo 2: Filme Irradiado")
    arquivo_irrad = st.file_uploader("Envie o filme COM radiação", 
                                     type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
                                     key="irrad")
    
    if arquivo_irrad:
        img_irrad = Image.open(io.BytesIO(arquivo_irrad.read()))
        if img_irrad.mode != 'RGB':
            img_irrad = img_irrad.convert('RGB')
        img_irrad_np = np.array(img_irrad)
        img_irrad_norm = img_irrad_np / 255.0 if img_irrad_np.max() > 1 else img_irrad_np
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Filme Irradiado")
            st.image(img_irrad_np, use_column_width=True)
        
        if st.button("🔍 ANALISAR", type="primary", use_container_width=True):
            with st.spinner("Processando..."):
                
                # Detectar regiões
                regioes, limite = detectar_regioes(img_irrad_norm, media_ref, desvio_ref, fator_desvio, area_min)
                
                st.info(f"Limite de detecção: {limite:.4f} (Média - {fator_desvio}σ)")
                
                if not regioes:
                    st.warning("⚠️ Nenhuma região detectada!")
                    st.info("Tente reduzir 'Sensibilidade (σ)' ou 'Área Mínima'")
                    st.stop()
                
                # Ordenar e desenhar
                reg_ord = ordenar(regioes)
                img_res = desenhar(img_irrad_norm, reg_ord)
                
                with col4:
                    st.subheader(f"Resultado: {len(regioes)} regiões")
                    st.image(img_res, use_column_width=True)
                
                # Tabela
                st.markdown("---")
                st.header("📊 Dados das Regiões")
                
                mm = 25.4 / dpi
                df = pd.DataFrame([{
                    'N': r['id_ordenado'],
                    'Area_mm2': round(r['area'] * mm**2, 2),
                    'Intensidade': round(r['intensidade_media'], 4)
                } for r in reg_ord])
                
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Download
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "resultado.csv", "text/csv")
else:
    st.info("👆 Primeiro envie o filme base (não irradiado)")
