import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, square, erosion, dilation, opening
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

def detectar_regioes_global(imagem, area_min, offset, fechamento, erosao):
    """Detecta regioes com parametros globais"""
    gray = rgb2gray(imagem)
    thresh = threshold_otsu(gray)
    thresh_ajustado = thresh * (1 - offset)
    binary = gray < thresh_ajustado
    
    if erosao > 0:
        binary = erosion(binary, square(erosao))
    binary = remove_small_objects(binary, min_size=area_min)
    if fechamento > 0:
        binary = closing(binary, square(fechamento))
    
    labeled = label(binary)
    regions = regionprops(labeled, intensity_image=gray)
    
    regioes = []
    for i, r in enumerate(regions):
        if r.area >= area_min:
            minr, minc, maxr, maxc = r.bbox
            w = maxc - minc
            h = maxr - minr
            razao = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            regioes.append({
                'id': i,
                'area': r.area,
                'intensidade': r.mean_intensity,
                'centro': (int(r.centroid[1]), int(r.centroid[0])),
                'bbox': (minc, minr, w, h),
                'razao': razao,
                'coords': r.coords
            })
    
    return regioes, thresh, thresh_ajustado, binary, gray

def ajustar_regiao(indice, gray, regiao, erosao_ind, dilatacao_ind, fechamento_ind):
    """Ajusta uma regiao individual"""
    # Criar mascara para esta regiao
    mascara = np.zeros(gray.shape, dtype=bool)
    for coord in regiao['coords']:
        mascara[coord[0], coord[1]] = True
    
    # Aplicar operacoes morfologicas individuais
    if erosao_ind > 0:
        mascara = erosion(mascara, square(erosao_ind))
    if dilatacao_ind > 0:
        mascara = dilation(mascara, square(dilatacao_ind))
    if fechamento_ind > 0:
        mascara = closing(mascara, square(fechamento_ind))
    
    # Recalcular propriedades
    labeled = label(mascara)
    regions = regionprops(labeled, intensity_image=gray)
    
    if len(regions) > 0:
        r = max(regions, key=lambda x: x.area)
        minr, minc, maxr, maxc = r.bbox
        w = maxc - minc
        h = maxr - minr
        return {
            'id': regiao['id'],
            'area': r.area,
            'intensidade': r.mean_intensity,
            'centro': (int(r.centroid[1]), int(r.centroid[0])),
            'bbox': (minc, minr, w, h),
            'razao': min(w, h) / max(w, h) if max(w, h) > 0 else 0
        }
    return regiao

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

# Interface
with st.sidebar:
    st.header("Configuracoes Globais")
    dpi = st.number_input("DPI", 1, 2400, 50)
    area_min = st.slider("Area Minima (pixels)", 100, 50000, 3000, 100)
    offset = st.slider("Sensibilidade", 0.0, 0.5, 0.15, 0.01)
    fechamento = st.slider("Fechamento Global", 0, 20, 5, 1)
    erosao = st.slider("Erosao Global", 0, 10, 0, 1)

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
            
            regioes, thresh_orig, thresh_ajust, binary, gray = detectar_regioes_global(
                img_filme, area_min, offset, fechamento, erosao)
            
            st.info(f"Threshold: {thresh_orig:.3f} -> {thresh_ajust:.3f} | {len(regioes)} regioes detectadas")
            
            if not regioes:
                st.warning("Nenhuma regiao! Aumente Sensibilidade")
                st.stop()
            
            # Salvar no session state para ajuste individual
            st.session_state['regioes'] = regioes
            st.session_state['gray'] = gray
            st.session_state['img_filme'] = img_filme
            st.session_state['ajustado'] = False
    
    # Ajuste individual das regioes
    if 'regioes' in st.session_state:
        st.markdown("---")
        st.header("Ajuste Individual das Regioes")
        st.info("Ajuste cada regiao separadamente se necessario")
        
        regioes = st.session_state['regioes']
        gray = st.session_state['gray']
        img_filme = st.session_state['img_filme']
        
        regioes_ajustadas = []
        
        cols = st.columns(len(regioes))
        for i, (col, regiao) in enumerate(zip(cols, regioes)):
            with col:
                st.markdown(f"**Regiao {i+1}**")
                erosao_ind = st.slider(f"Erosao R{i+1}", 0, 10, 0, 1, key=f"er_{i}")
                dilatacao_ind = st.slider(f"Dilatacao R{i+1}", 0, 10, 0, 1, key=f"di_{i}")
                fechamento_ind = st.slider(f"Fechamento R{i+1}", 0, 10, 0, 1, key=f"fe_{i}")
                
                # Aplicar ajuste
                reg_ajust = ajustar_regiao(i, gray, regiao, erosao_ind, dilatacao_ind, fechamento_ind)
                regioes_ajustadas.append(reg_ajust)
        
        # Ordenar e desenhar resultado final
        reg_ord = ordenar(regioes_ajustadas)
        img_res = desenhar(img_filme, reg_ord)
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Resultado Final")
            st.image(img_res, use_column_width=True)
        
        # Tabela
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
