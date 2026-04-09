"""
Project Analyzer - Baseado no grain-analysis de José Henrique Roveda
https://github.com/josehenriqueroveda/grain-analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import cv2

st.set_page_config(page_title="Project Analyzer", page_icon="🔬", layout="wide")

st.title("🔬 Project Analyzer")
st.markdown("Detecção de regiões irradiadas em filme EBT3")

def detectar_fundo(imagem):
    """Detecta e corta o filme da imagem"""
    gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    
    # Otsu para separar fundo branco do filme
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return imagem, None
    
    # Pegar o maior contorno (o filme)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Cortar com margem
    margem = 20
    h_img, w_img = imagem.shape[:2]
    x1 = max(0, x - margem)
    y1 = max(0, y - margem)
    x2 = min(w_img, x + w + margem)
    y2 = min(h_img, y + h + margem)
    
    return imagem[y1:y2, x1:x2], (x1, y1, x2, y2)

def detectar_regioes(imagem, area_min_pixels):
    """
    Detecta regiões irradiadas usando técnica do grain-analysis
    """
    # Converter para grayscale
    gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    
    # Blur para reduzir ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold Otsu (inverso - detecta escuro)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Processar contornos
    regioes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_min_pixels:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calcular intensidade média da região
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            intensidade = cv2.mean(gray, mask=mask)[0]
            
            # Centro
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            regioes.append({
                'area': area,
                'intensidade_media': intensidade,
                'centro': (cx, cy),
                'bbox': (x, y, w, h),
                'contorno': contour
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
    img = imagem.copy()
    
    for r in regioes:
        x, y, w, h = r['bbox']
        
        # Contorno verde
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Número
        cx, cy = r['centro']
        numero = str(r['id_ordenado'])
        
        # Fundo preto
        (tw, th), _ = cv2.getTextSize(numero, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(img, (cx - tw//2 - 5, cy - th//2 - 5), 
                     (cx + tw//2 + 5, cy + th//2 + 5), (0, 0, 0), -1)
        
        # Texto branco
        cv2.putText(img, numero, (cx - tw//2, cy + th//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

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
            'Intensidade Média': round(r['intensidade_media'], 2),
            'Centro X': r['centro'][0],
            'Centro Y': r['centro'][1]
        })
    return pd.DataFrame(dados)

# Interface
with st.sidebar:
    st.header("⚙️ Configurações")
    dpi = st.number_input("DPI do Scanner", 1, 2400, 50)
    area_min = st.slider("Área Mínima (pixels)", 100, 10000, 1000, 100, 
                        help="Aumente se detectar ruídos pequenos")
    st.markdown("---")
    st.info("💡 Baseado em grain-analysis\ngithub.com/josehenriqueroveda/grain-analysis")

st.header("📁 Upload da Imagem")
arquivo = st.file_uploader("Selecione a imagem do filme EBT3", 
                          type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])

if arquivo:
    # Ler imagem
    img_pil = Image.open(io.BytesIO(arquivo.read()))
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    img_original = np.array(img_pil)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(img_original, use_column_width=True)
    
    if st.button("🔍 ANALISAR", type="primary", use_container_width=True):
        with st.spinner("Processando..."):
            
            # Passo 1: Detectar filme
            img_filme, bbox = detectar_fundo(img_original)
            
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
