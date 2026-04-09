"""
Project Analyzer v3.0
Baseado no código do Dosepy (Luis Olivares)
https://github.com/LuisOlivaresJ/Dosepy
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

# Importações do Dosepy
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation, remove_small_objects, remove_small_holes
from skimage.morphology import square
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, find_boundaries

st.set_page_config(
    page_title="Project Analyzer v3",
    page_icon="🔬",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def detectar_fundo_dosepy(imagem):
    """
    Detecta o filme usando técnica EXATA do Dosepy:
    - Otsu + erosão + label
    """
    gray = rgb2gray(imagem)
    
    # Threshold Otsu (igual ao Dosepy)
    thresh = threshold_otsu(gray)
    binary = gray < thresh  # Filme é mais escuro que fundo
    
    # Limpar bordas
    binary = clear_border(binary)
    
    # Erosão para limpar bordas (técnica do Dosepy)
    binary = erosion(binary, square(5))
    binary = dilation(binary, square(3))
    
    # Remover ruídos
    binary = remove_small_objects(binary, min_size=1000)
    binary = remove_small_holes(binary, area_threshold=1000)
    
    # Labeling (igual ao Dosepy)
    labeled = label(binary)
    regions = regionprops(labeled)
    
    if not regions:
        return imagem, None
    
    # Pegar a maior região (o filme) - igual ao Dosepy
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

def detectar_regioes_dosepy(imagem, area_min_percent=0.1, thresh_percent=80):
    """
    Detecta regiões irradiadas DENTRO do filme
    Adaptado do Dosepy - detecta múltiplos níveis de escurecimento
    """
    gray = rgb2gray(imagem)
    
    # Área mínima em pixels
    area_total = imagem.shape[0] * imagem.shape[1]
    area_minima = (area_min_percent / 100) * area_total
    
    # Calcular a intensidade média do filme (região não irradiada)
    # Usamos a moda (valor mais frequente) como referência
    hist, bin_edges = np.histogram(gray.flatten(), bins=256, range=(0, 1))
    modo_idx = np.argmax(hist)
    filme_base = (bin_edges[modo_idx] + bin_edges[modo_idx + 1]) / 2
    
    # Detectar TODAS as regiões mais escuras que o filme base
    # thresh_percent: porcentagem do escurecimento mínimo para considerar
    # 80% = detecta tudo que é pelo menos 20% mais escuro que o filme base
    limite = filme_base * (thresh_percent / 100)
    
    binary = gray < limite
    
    # Limpeza morfológica (igual ao Dosepy)
    binary = remove_small_objects(binary, min_size=int(area_minima))
    binary = remove_small_holes(binary, area_threshold=int(area_minima/2))
    binary = erosion(binary, square(3))
    binary = dilation(binary, square(2))
    
    # Labeling (igual ao Dosepy)
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray)
    
    # Filtrar por área mínima
    regioes = []
    for i, region in enumerate(regions):
        if region.area < area_minima:
            continue
        
        regioes.append({
            'id': i,
            'area': region.area,
            'intensidade_media': region.mean_intensity,
            'centro': (int(region.centroid[1]), int(region.centroid[0])),
            'bbox': region.bbox,
        })
    
    return regioes

def ordenar_por_escurecimento(regioes):
    """
    Ordena do mais CLARO (maior intensidade) para o mais ESCURO (menor intensidade)
    """
    regioes_ordenadas = sorted(regioes, key=lambda x: x['intensidade_media'], reverse=True)
    
    for i, regiao in enumerate(regioes_ordenadas, 1):
        regiao['id_ordenado'] = i
    
    return regioes_ordenadas

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
    
    for regiao in regioes:
        minr, minc, maxr, maxc = regiao['bbox']
        
        # Contorno verde
        draw.rectangle([minc, minr, maxc, maxr], outline=(0, 255, 0), width=3)
        
        # Número no centro
        cx, cy = regiao['centro']
        numero = str(regiao['id_ordenado'])
        
        bbox = draw.textbbox((0, 0), numero, font=fonte)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([cx-tw//2-4, cy-th//2-4, cx+tw//2+4, cy+th//2+4], fill=(0, 0, 0))
        draw.text((cx-tw//2, cy-th//2), numero, fill=(255, 255, 255), font=fonte)
    
    return np.array(img_pil)

def criar_dataframe(regioes, dpi=50):
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

# ============ INTERFACE ============

st.markdown('<p class="main-title">🔬 Project Analyzer v3.0</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Baseado no Dosepy de Luis Olivares</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    dpi = st.number_input("DPI do Scanner", 1, 2400, 50)
    
    area_min = st.slider(
        "Área Mínima (%)", 
        0.01, 5.0, 0.5, 0.01,
        help="Regiões menores são ignoradas como ruído"
    )
    
    thresh_percent = st.slider(
        "Threshold (% do filme base)",
        50, 99, 80, 1,
        help="80% = detecta tudo 20% mais escuro que o filme base"
    )
    
    st.markdown("---")
    st.info("💡 Baseado no Dosepy\ngithub.com/LuisOlivaresJ/Dosepy")

# Upload
st.header("📁 Upload da Imagem")
arquivo = st.file_uploader(
    "Selecione a imagem do filme EBT3",
    type=['tif', 'tiff', 'png', 'jpg', 'jpeg']
)

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
        with st.spinner("Processando com Dosepy..."):
            
            # Passo 1: Detectar filme (código Dosepy)
            img_filme, bbox = detectar_fundo_dosepy(img_normalized)
            
            if bbox is None:
                st.error("❌ Não foi possível detectar o filme!")
                st.stop()
            
            st.info(f"✅ Filme detectado: {img_filme.shape[1]}x{img_filme.shape[0]} px")
            
            # Passo 2: Detectar regiões irradiadas (adaptado do Dosepy)
            regioes = detectar_regioes_dosepy(img_filme, area_min, thresh_percent)
            
            if not regioes:
                st.warning("⚠️ Nenhuma região detectada!")
                st.info("Tente reduzir 'Área Mínima' ou aumentar 'Threshold'")
                st.stop()
            
            # Passo 3: Ordenar
            regioes_ord = ordenar_por_escurecimento(regioes)
            
            # Passo 4: Desenhar
            img_resultado = desenhar_resultado(img_filme, regioes_ord)
            
            # DataFrame
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
            
            # Tabela
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
                Image.fromarray(img_resultado).save(buf, format='PNG')
                st.download_button("📥 Download Imagem", buf.getvalue(),
                                 f"analisado_{arquivo.name.split('.')[0]}.png",
                                 "image/png", use_container_width=True)

else:
    st.info("👆 Faça upload de uma imagem para começar!")
l)
                """)
                
                # Download do CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"resultado_{arquivo.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
                
                # Download da imagem
                resultado_pil = Image.fromarray(imagem_resultado)
                buf = io.BytesIO()
                resultado_pil.save(buf, format='PNG')
                byte_imagem = buf.getvalue()
                
                st.download_button(
                    label="⬇️ Download Imagem",
                    data=byte_imagem,
                    file_name=f"analisado_{arquivo.name.split('.')[0]}.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
