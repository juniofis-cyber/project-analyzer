"""
Project Analyzer - Baseado no grain-analysis (adaptado com skimage)
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import closing, square, remove_small_objects
from skimage.measure import label, regionprops, find_contours
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
    
    # Remover ruídos pequenos
    binary = remove_small_objects(binary, min_size=1000)
    
    # Labeling
    labeled = label(binary)
    regions = regionprops(labeled)
    
    if not regions:
        return imagem, None
    
    # Pegar o maior contorno (o filme)
    largest = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox
    
    # Cortar com margem
    margem = 20
    h, w = imagem.shape[:2]
    minr = max(0, minr - margem)
    minc = max(0, minc - margem)
    maxr = min(h, maxr + margem)
    maxc = min(w, maxc + margem)
    
    return imagem[minr:maxr, minc:maxc], (minr, minc, maxr, maxc)

def detectar_regioes(imagem, area_min_pixels):
    """
    Detecta regiões irradiadas usando técnica similar ao grain-analysis
    mas com skimage
    """
    gray = rgb2gray(imagem)
    
    # Blur para reduzir ruído
    blurred = gaussian(gray, sigma=2)
    
    # Threshold Otsu (inverso - detecta escuro)
    thresh = threshold_otsu(blurred)
    binary = blurred < thresh
    
    # Remover objetos pequenos
    binary = remove_small_objects(binary, min_size=int(area_min_pixels))
    
    # Fechar buracos
    binary = closing(binary, square(5))
    
    # Labeling
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray)
    
    # Processar regiões
    regioes = []
    for region in regions:
        if region.area < area_min_pixels:
            continue
        
        minr, minc, maxr, maxc = region.bbox
        w = maxc - minc
        h = maxr - minr
        
        # Calcular intensidade média
        intensidade = region.mean_intensity
        
        # Centro
        cx = int(region.centroid[1])
        cy = int(region.centroid[0])
        
        regioes.append({
            'area': region.area,
            'intensidade_media': intensidade,
            'centro': (cx, cy),
            'bbox': (minc, minr, w, h)
        })
    
    return regioes

def ordenar_por_escurecimento(regioes):
    """Ordena do mais claro (maior intensidade) para o mais escuro"""
    ordenadas = sorted(regioes, key=lambda x: x['intensidade_media'], reverse=True)
    for i, r in enumerate(ordenadas, 1):
        r['id_ordenado'] = i
    return ordenadas

def desenhar_resultado(imagem, regioes):
    """Desenha contornos e numeração"""
    # Converter para uint8
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
    
    for r in regioes:
        x, y, w, h = r['bbox']
        
        # Contorno verde
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
        
        # Número no centro
        cx, cy = r['centro']
        numero = str(r['id_ordenado'])
        
        # Fundo preto
        bbox = draw.textbbox((0, 0), numero, font=fonte)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([cx - tw//2 - 4, cy - th//2 - 4, 
                       cx + tw//2 + 4, cy + th//2 + 4], fill=(0, 0, 0))
        
        # Texto branco
        draw.text((cx - tw//2, cy - th//2), numero, fill=(255, 255, 255), font=fonte)
    
    return np.array(img_pil)

def criar_dataframe(regioes, dpi):
    """Cria DataFrame com dados"""
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
