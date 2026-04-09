import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import closing, square, remove_small_objects
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border

st.set_page_config(page_title="Project Analyzer", page_icon="🔬", layout="wide")

st.title("🔬 Project Analyzer")
st.markdown("Detecção de regiões irradiadas em filme EBT3")

def detectar_fundo(imagem):
    """Detecta e corta o filme"""
    gray = rgb2gray(imagem)
    thresh = threshold_otsu(gray)
    binary = gray < thresh
    binary = clear_border(binary)
    binary = remove_small_objects(binary, min_size=1000)
    labeled = label(binary)
    regions = regionprops(labeled)
    if not regions:
        return imagem, None
    largest = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox
    margem = 20
    h, w = imagem.shape[:2]
    return imagem[max(0,minr-margem):min(h,maxr+margem), max(0,minc-margem):min(w,maxc+margem)], True

def detectar_regioes_bordas(imagem, area_min_pct):
    """Detecta regiões usando detecção de bordas Canny"""
    gray = rgb2gray(imagem)
    area_total = imagem.shape[0] * imagem.shape[1]
    area_minima = (area_min_pct / 100) * area_total
    
    # Detectar bordas com Canny
    bordas = canny(gray, sigma=2)
    
    # Fechar bordas para formar regiões
    bordas_fechadas = closing(bordas, square(5))
    
    # Preencher regiões
    labeled = label(bordas_fechadas)
    regions = regionprops(labeled, intensity_image=gray)
    
    # Filtrar por área e forma (quadrados/retângulos)
    regioes = []
    for r in regions:
        if r.area < area_minima:
            continue
        # Calcular razão de aspecto (quadrados têm razão próxima de 1)
        minr, minc, maxr, maxc = r.bbox
        altura = maxr - minr
        largura = maxc - minc
        if altura > 0 and largura > 0:
            razao = min(altura, largura) / max(altura, largura)
            # Aceitar razões entre 0.5 e 1.0 (quadrados a retângulos)
            if razao >= 0.5:
                regioes.append({
                    'area': r.area,
                    'intensidade_media': r.mean_intensity,
                    'centro': (int(r.centroid[1]), int(r.centroid[0])),
                    'bbox': r.bbox,
                    'razao': razao
                })
    
    return regioes

def detectar_regioes_otsu_local(imagem, area_min_pct, fator_escuro):
    """Detecta regiões mais escuras que o filme base"""
    gray = rgb2gray(imagem)
    area_total = imagem.shape[0] * imagem.shape[1]
    area_minima = (area_min_pct / 100) * area_total
    
    # Calcular histograma para encontrar o pico (filme base)
    hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 1))
    pico_idx = np.argmax(hist)
    pico_valor = (bins[pico_idx] + bins[pico_idx + 1]) / 2
    
    # Detectar tudo que é significativamente mais escuro que o pico
    limite = pico_valor * (1 - fator_escuro)
    binary = gray < limite
    
    # Limpeza
    binary = remove_small_objects(binary, min_size=int(area_minima))
    binary = closing(binary, square(5))
    
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray)
    
    regioes = []
    for r in regions:
        if r.area < area_minima:
            continue
        minr, minc, maxr, maxc = r.bbox
        altura = maxr - minr
        largura = maxc - minc
        if altura > 0 and largura > 0:
            razao = min(altura, largura) / max(altura, largura)
            if razao >= 0.4:  # Um pouco mais flexível
                regioes.append({
                    'area': r.area,
                    'intensidade_media': r.mean_intensity,
                    'centro': (int(r.centroid[1]), int(r.centroid[0])),
                    'bbox': r.bbox,
                    'razao': razao
                })
    
    return regioes, pico_valor, limite

def ordenar(regioes):
    ordenadas = sorted(regioes, key=lambda x: x['intensidade_media'], reverse=True)
    for i, r in enumerate(ordenadas, 1):
        r['id_ordenado'] = i
    return ordenadas

def desenhar(imagem, regioes):
    img = Image.fromarray((imagem * 255).astype(np.uint8) if imagem.max() <= 1 else imagem.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    try:
        fonte = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
    except:
        fonte = ImageFont.load_default()
    for r in regioes:
        minr, minc, maxr, maxc = r['bbox']
        draw.rectangle([minc, minr, maxc, maxr], outline=(0, 255, 0), width=3)
        cx, cy = r['centro']
        bbox = draw.textbbox((0, 0), str(r['id_ordenado']), font=fonte)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([cx-tw//2-4, cy-th//2-4, cx+tw//2+4, cy+th//2+4], fill=(0, 0, 0))
        draw.text((cx-tw//2, cy-th//2), str(r['id_ordenado']), fill=(255, 255, 255), font=fonte)
    return np.array(img)

def criar_df(regioes, dpi):
    mm = 25.4 / dpi
    return pd.DataFrame([{'Número': r['id_ordenado'], 'Área (mm²)': round(r['area']*mm**2, 2), 'Intensidade': round(r['intensidade_media'], 4), 'Razão': round(r.get('razao', 0), 2)} for r in regioes])

with st.sidebar:
    st.header("Configurações")
    dpi = st.number_input("DPI", 1, 2400, 50)
    area_min = st.slider("Área Mínima (%)", 0.01, 10.0, 5.0, 0.1, help="Aumente se detectar áreas muito pequenas")
    fator_escuro = st.slider("Fator de Escurecimento", 0.01, 0.30, 0.08, 0.01, help="0.08 = detecta 8% mais escuro que o filme base")
    metodo = st.selectbox("Método", ["Otsu Local", "Bordas Canny"])

arquivo = st.file_uploader("Upload da imagem EBT3", type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])

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
            img_filme, ok = detectar_fundo(img_norm)
            if not ok:
                st.error("Filme não detectado!")
                st.stop()
            
            if metodo == "Otsu Local":
                regioes, pico, limite = detectar_regioes_otsu_local(img_filme, area_min, fator_escuro)
                st.info(f"Filme base: {pico:.3f} | Limite: {limite:.3f}")
            else:
                regioes = detectar_regioes_bordas(img_filme, area_min)
            
            if not regioes:
                st.warning("Nenhuma região! Aumente 'Área Mínima' ou 'Fator de Escurecimento'")
                st.stop()
            
            reg_ord = ordenar(regioes)
            img_res = desenhar(img_filme, reg_ord)
            df = criar_df(reg_ord, dpi)
            
            with col2:
                st.subheader(f"{len(regioes)} regiões")
                st.image(img_res, use_column_width=True)
            
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "resultado.csv", "text/csv")
else:
    st.info("Faça upload de uma imagem")
