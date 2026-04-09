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
    
    # thresholds[0] = separa fundo do filme
    # thresholds[1] = separa filme das regiões irradiadas
    
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
                'Nº': r['id_ordenado'],
                'Área (mm²)': round(r['area'] * mm**2, 2),
                'Intensidade': round(r['intensidade_media'], 4)
            } for r in reg_ord])
            
            st.dataframe(df, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "resultado.csv", "text/csv")
else:
    st.info("Faça upload de uma imagem")
    'Área (mm²)': round(area_mm2, 2),
            'Área (pixels)': int(r['area']),
            'Intensidade Média': round(r['intensidade_media'], 4),
            'Centro X': r['centro'][0],
            'Centro Y': r['centro'][1]
        })
    return pd.DataFrame(dados)

# Interface
with st.sidebar:
    st.header("⚙️ Configurações")
    dpi = st.number_input("DPI do Scanner", 1, 2400, 50)
    area_min = st.slider("Área Mínima (pixels)", 100, 20000, 5000, 100, 
                        help="Aumente se detectar ruídos pequenos")
    st.markdown("---")
    st.info("💡 Baseado em grain-analysis")

st.header("📁 Upload da Imagem")
arquivo = st.file_uploader("Selecione a imagem do filme EBT3", 
                          type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])

if arquivo:
    # Ler imagem
    img_pil = Image.open(io.BytesIO(arquivo.read()))
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    img_original = np.array(img_pil)
    
    # Normalizar
    if img_original.max() > 1:
        img_normalized = img_original / 255.0
    else:
        img_normalized = img_original
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(img_original, use_column_width=True)
    
    if st.button("🔍 ANALISAR", type="primary", use_container_width=True):
        with st.spinner("Processando..."):
            
            # Passo 1: Detectar filme
            img_filme, bbox = detectar_fundo(img_normalized)
            
            if bbox is None:
                st.error("❌ Não foi possível detectar o filme!")
                st.stop()
            
            st.info(f"✅ Filme detectado: {img_filme.shape[1]}x{img_filme.shape[0]} pixels")
            
            # Passo 2: Detectar regiões
            regioes = detectar_regioes(img_filme, area_min)
            
            if not regioes:
                st.warning("⚠️ Nenhuma região detectada!")
                st.info("Tente reduzir 'Área Mínima'")
                st.stop()
            
            # Passo 3: Ordenar
            regioes_ord = ordenar_por_escurecimento(regioes)
            
            # Passo 4: Desenhar
            img_resultado = desenhar_resultado(img_filme, regioes_ord)
            df = criar_dataframe(regioes_ord, dpi)
            
            # Mostrar resultado
            with col2:
                st.subheader(f"Resultado: {len(regioes)} regiões")
                st.image(img_resultado, use_column_width=True)
            
            # Métricas
            st.markdown("---")
            st.header("📊 Análise")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Regiões", len(regioes))
            c2.metric("DPI", dpi)
            c3.metric("Resolução", f"{25.4/dpi:.2f} mm/px")
            c4.metric("Ordenação", "1=Claro → N=Escuro")
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Downloads
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, 
                                 f"resultado_{arquivo.name.split('.')[0]}.csv", 
                                 "text/csv", use_container_width=True)
            
            with col_d2:
                buf = io.BytesIO()
                img_pil_res = Image.fromarray(img_resultado)
                img_pil_res.save(buf, format='PNG')
                st.download_button("📥 Download Imagem", buf.getvalue(),
                                 f"analisado_{arquivo.name.split('.')[0]}.png",
                                 "image/png", use_container_width=True)
else:
    st.info("👆 Faça upload de uma imagem para começar!")
