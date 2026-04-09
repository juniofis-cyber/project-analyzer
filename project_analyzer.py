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
st.title("🔬 Project Analyzer")

def cortar_filme(imagem):
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

def detectar_regioes(imagem, area_min, offset):
    gray = rgb2gray(imagem)
    thresh = threshold_otsu(gray)
    thresh_ajustado = thresh * (1 - offset)
    binary = gray < thresh_ajustado
    binary = remove_small_objects(binary, min_size=area_min)
    binary = closing(binary, square(5))
    binary = erosion(binary, square(3))
    binary = dilation(binary, square(3))
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray)
    regioes = []
    for r in regions:
        if r.area >= area_min:
            minr, minc, maxr, maxc = r.bbox
            w = maxc - minc
            h = maxr - minr
            razao = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            regioes.append({
                'area': r.area,
                'intensidade': r.mean_intensity,
                'centro': (int(r.centroid[1]), int(r.centroid[0])),
                'bbox': (minc, minr, w, h),
                'razao': razao
            })
    return regioes, thresh, thresh_ajustado

def ordenar(regioes):
    ordenadas = sorted(regioes, key=lambda x: x['intensidade'], reverse=True)
    for i, r in enumerate(ordenadas, 1):
        r['id'] = i
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
        draw.rectangle([x, y, x+w, y+h], outline=(0, 255, 0), width=3)
        cx, cy = r['centro']
        txt = str(r['id'])
        bbox = draw.textbbox((0, 0), txt, font=fonte)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.rectangle([cx-tw//2-4, cy-th//2-4, cx+tw//2+4, cy+th//2+4], fill=(0,0,0))
        draw.text((cx-tw//2, cy-th//2), txt, fill=(255,255,255), font=fonte)
    return np.array(img_pil)

with st.sidebar:
    st.header("Configuracoes")
    dpi = st.number_input("DPI", 1, 2400, 50)
    area_min = st.slider("Area Minima", 100, 50000, 3000, 100)
    offset = st.slider("Sensibilidade", 0.0, 0.5, 0.15, 0.01)

st.header("Upload da Imagem")
arquivo = st.file_uploader("Imagem EBT3", type=['tif','tiff','png','jpg','jpeg'])

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
    
    if st.button("ANALISAR", type="primary"):
        with st.spinner("Processando..."):
            img_filme = cortar_filme(img_norm)
            if img_filme is None:
                st.error("Filme nao detectado!")
                st.stop()
            st.info(f"Filme: {img_filme.shape[1]}x{img_filme.shape[0]} px")
            
            regioes, thresh_orig, thresh_ajust = detectar_regioes(img_filme, area_min, offset)
            st.info(f"Threshold: {thresh_orig:.3f} -> {thresh_ajust:.3f}")
            
            if not regioes:
                st.warning("Nenhuma regiao! Aumente Sensibilidade")
                st.stop()
            
            reg_ord = ordenar(regioes)
            img_res = desenhar(img_filme, reg_ord)
            
            with col2:
                st.subheader(f"{len(regioes)} regioes")
                st.image(img_res, use_column_width=True)
            
            mm = 25.4 / dpi
            df = pd.DataFrame([{
                'N': r['id'],
                'Area_mm2': round(r['area'] * mm * mm, 2),
                'Intensidade': round(r['intensidade'], 4),
                'Razao': round(r['razao'], 2)
            } for r in reg_ord])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "resultado.csv", "text/csv")
else:
    st.info("Faca upload de uma imagem")
