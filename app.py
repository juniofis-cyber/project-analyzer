import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import json
from pathlib import Path

from image_proc import (
    read_image,
    get_red_channel,
    normalize_to_16bit,
    compute_net_optical_density,
    average_roi,
)
from calibration import (
    CalibrationData,
    CalibrationModel,
    fit_calibration,
    predict_dose,
    save_model,
    load_model,
    model_to_dict,
)

st.set_page_config(
    page_title="EBT3/EBT4 — Análise de Filmes Radiocrômicos",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("☢️ Menu Principal")
app_mode = st.sidebar.radio(
    "Escolha a etapa:",
    ["📊 Calibração", "🎯 Análise de Dose", "📖 Guia Rápido"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Filmes suportados:** EBT3 / EBT4\n\n"
    "**Canal analisado:** Vermelho (16-bit preservado)\n\n"
    "**Métrica:** Net Optical Density (NOD)"
)

# ============================================================
# ABA 1: CALIBRAÇÃO
# ============================================================
if app_mode == "📊 Calibração":
    st.title("📊 Curva de Calibração EBT3 / EBT4")
    st.markdown(
        """
        Crie a curva de calibração associando **Net Optical Density (NOD)** à **Dose (Gy)**.
        
        **Passos:**
        1. Faça upload da imagem de controle (filme não irradiado)
        2. Faça upload das imagens irradiadas com doses conhecidas
        3. Selecione uma ROI em cada imagem para extrair a média de pixel
        4. Informe as doses correspondentes
        5. Escolha o modelo matemático e ajuste a curva
        """
    )

    col_ctrl, col_imgs = st.columns(2)

    with col_ctrl:
        st.subheader("1. Imagem de Controle (Background)")
        ctrl_file = st.file_uploader(
            "Upload do filme NÃO irradiado", type=["tif", "tiff", "png", "jpg", "jpeg"]
        )
        if ctrl_file:
            ctrl_bytes = ctrl_file.read()
            ctrl_img = read_image(io.BytesIO(ctrl_bytes))
            ctrl_red = get_red_channel(ctrl_img, bgr_format=True)
            ctrl_red = normalize_to_16bit(ctrl_red)
            st.success(f"Controle carregado: shape {ctrl_img.shape}, tipo {ctrl_img.dtype}")
            st.image(ctrl_img, caption="Imagem de controle", use_container_width=True)

    with col_imgs:
        st.subheader("2. Imagens Irradiadas")
        dose_files = st.file_uploader(
            "Upload dos filmes irradiados (doses conhecidas)",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

    if ctrl_file and dose_files:
        st.subheader("3. Extração de NOD por amostra")

        n_samples = len(dose_files)
        dose_values_input = st.text_input(
            f"Informe as {n_samples} doses em Gy (separadas por vírgula)",
            value=", ".join([str((i + 1) * 2.0) for i in range(n_samples)]),
            help="Exemplo: 0.5, 1.0, 2.0, 5.0, 10.0",
        )

        try:
            dose_values = [float(x.strip()) for x in dose_values_input.split(",")]
            if len(dose_values) != n_samples:
                st.error(f"Número de doses ({len(dose_values)}) não bate com número de imagens ({n_samples})")
                dose_values = None
        except ValueError:
            st.error("Formato inválido. Use números separados por vírgula.")
            dose_values = None

        if dose_values:
            nod_values = []

            cols = st.columns(min(n_samples, 4))
            for idx, (dose_file, dose_gy) in enumerate(zip(dose_files, dose_values)):
                with cols[idx % 4]:
                    st.markdown(f"**Amostra {idx + 1}: {dose_gy} Gy**")
                    dose_bytes = dose_file.read()
                    dose_img = read_image(io.BytesIO(dose_bytes))
                    dose_red = get_red_channel(dose_img, bgr_format=True)
                    dose_red = normalize_to_16bit(dose_red)

                    # ROI automática central (20% da imagem)
                    h, w = dose_red.shape
                    roi_w, roi_h = max(20, int(w * 0.2)), max(20, int(h * 0.2))
                    roi_x, roi_y = (w - roi_w) // 2, (h - roi_h) // 2

                    st.image(dose_img, caption=f"{dose_file.name}", use_container_width=True)

                    # Cálculo do NOD
                    nod_map = compute_net_optical_density(dose_red, ctrl_red, bit_depth=16)
                    avg_nod = average_roi(nod_map, roi_x, roi_y, roi_w, roi_h)
                    nod_values.append(avg_nod)

                    st.metric("NOD médio (ROI central)", f"{avg_nod:.4f}")

            if nod_values:
                st.subheader("4. Ajuste da Curva de Calibração")

                col_model, col_viz = st.columns([1, 2])

                with col_model:
                    model_type = st.selectbox(
                        "Modelo matemático:",
                        ["power_law", "polynomial3", "polynomial2", "rational", "log_linear", "spline"],
                        index=0,
                    )
                    model_name_map = {
                        "power_law": "Power Law: Dose = a·NODᵇ",
                        "polynomial3": "Polinômio 3º grau",
                        "polynomial2": "Polinômio 2º grau",
                        "rational": "Racional: (a+b·NOD)/(1+c·NOD)",
                        "log_linear": "Log-linear: exp(a + b·ln(NOD))",
                        "spline": "Spline cúbica (interpolação)",
                    }
                    st.caption(model_name_map[model_type])

                    if st.button("🔧 Ajustar Curva", type="primary"):
                        data = CalibrationData(
                            nod=np.array(nod_values, dtype=np.float64),
                            dose=np.array(dose_values, dtype=np.float64),
                        )
                        model = fit_calibration(data, model_type=model_type)

                        st.session_state["calibration_model"] = model
                        st.session_state["calibration_data"] = data
                        st.success(
                            f"✅ Modelo ajustado!\n\n"
                            f"R² = {model.r_squared:.4f} | RMSE = {model.rmse:.4f} Gy"
                        )

                with col_viz:
                    if "calibration_model" in st.session_state:
                        model: CalibrationModel = st.session_state["calibration_model"]
                        data: CalibrationData = st.session_state["calibration_data"]

                        fig, ax = plt.subplots(figsize=(8, 5))

                        # Pontos experimentais
                        ax.scatter(data.nod, data.dose, color="red", s=80, zorder=5, label="Pontos exp.")

                        # Curva ajustada
                        nod_fine = np.linspace(
                            max(0, np.min(data.nod) * 0.8),
                            np.max(data.nod) * 1.2,
                            500,
                        )
                        dose_pred = predict_dose(model, nod_fine)
                        ax.plot(nod_fine, dose_pred, "b-", linewidth=2, label=f"Ajuste ({model.model_type})")

                        ax.set_xlabel("Net Optical Density (NOD)")
                        ax.set_ylabel("Dose (Gy)")
                        ax.set_title(f"Curva de Calibração — R² = {model.r_squared:.4f}")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                        # Tabela de resíduos
                        st.markdown("**Resíduos:**")
                        dose_pred_pts = predict_dose(model, data.nod)
                        res_df = pd.DataFrame({
                            "Dose real (Gy)": data.dose,
                            "NOD": data.nod,
                            "Dose predita (Gy)": dose_pred_pts,
                            "Resíduo (Gy)": data.dose - dose_pred_pts,
                            "Resíduo (%)": 100 * (data.dose - dose_pred_pts) / np.where(data.dose > 0, data.dose, 1),
                        })
                        st.dataframe(res_df.style.format({
                            "Dose real (Gy)": "{:.3f}",
                            "NOD": "{:.4f}",
                            "Dose predita (Gy)": "{:.3f}",
                            "Resíduo (Gy)": "{:.3f}",
                            "Resíduo (%)": "{:.2f}",
                        }), use_container_width=True)

                        # Download do modelo
                        model_dict = model_to_dict(model)
                        model_json = json.dumps(model_dict, indent=2, ensure_ascii=False)
                        st.download_button(
                            "⬇️ Baixar modelo de calibração (JSON)",
                            data=model_json,
                            file_name="modelo_calibracao_ebt.json",
                            mime="application/json",
                        )


# ============================================================
# ABA 2: ANÁLISE DE DOSE
# ============================================================
elif app_mode == "🎯 Análise de Dose":
    st.title("🎯 Análise de Dose em Filme Desconhecido")
    st.markdown(
        """
        Aplique um modelo de calibração prévio para converter a imagem do filme
        em um **mapa de dose (Gy)**.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Carregar Modelo de Calibração")
        model_file = st.file_uploader("Upload do modelo JSON", type=["json"])
        if model_file:
            model_dict = json.load(model_file)
            model = CalibrationModel(
                model_type=model_dict["model_type"],
                params=np.array(model_dict["params"]) if model_dict["params"] else None,
                nod_min=model_dict["nod_min"],
                nod_max=model_dict["nod_max"],
                dose_min=model_dict["dose_min"],
                dose_max=model_dict["dose_max"],
                rmse=model_dict["rmse"],
                r_squared=model_dict["r_squared"],
            )
            st.session_state["analysis_model"] = model
            st.success(f"Modelo carregado: {model.model_type} (R²={model.r_squared:.4f})")

    with col2:
        st.subheader("2. Imagens para Análise")
        bg_file = st.file_uploader("Controle (não irradiado)", type=["tif", "tiff", "png", "jpg", "jpeg"])
        sample_file = st.file_uploader("Filme desconhecido", type=["tif", "tiff", "png", "jpg", "jpeg"])

    if "analysis_model" in st.session_state and bg_file and sample_file:
        model: CalibrationModel = st.session_state["analysis_model"]

        bg_img = read_image(io.BytesIO(bg_file.read()))
        sample_img = read_image(io.BytesIO(sample_file.read()))

        bg_red = get_red_channel(bg_img, bgr_format=True)
        bg_red = normalize_to_16bit(bg_red)
        sample_red = get_red_channel(sample_img, bgr_format=True)
        sample_red = normalize_to_16bit(sample_red)

        # Cálculo do NOD pixel a pixel
        nod_map = compute_net_optical_density(sample_red, bg_red, bit_depth=16)

        # Predição da dose
        dose_map = predict_dose(model, nod_map)

        st.subheader("3. Resultados")

        col_viz, col_stats = st.columns([3, 1])

        with col_viz:
            # Visualização lado a lado
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            im0 = axes[0].imshow(sample_img if sample_img.ndim == 3 else sample_img[:, :, 0], cmap="gray")
            axes[0].set_title("Filme original")
            axes[0].axis("off")

            im1 = axes[1].imshow(nod_map, cmap="viridis")
            axes[1].set_title("Net Optical Density (NOD)")
            plt.colorbar(im1, ax=axes[1], fraction=0.046)
            axes[1].axis("off")

            im2 = axes[2].imshow(dose_map, cmap="turbo")
            axes[2].set_title("Mapa de Dose (Gy)")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, label="Gy")
            axes[2].axis("off")

            plt.tight_layout()
            st.pyplot(fig)

        with col_stats:
            st.metric("Dose média", f"{np.mean(dose_map):.3f} Gy")
            st.metric("Dose mediana", f"{np.median(dose_map):.3f} Gy")
            st.metric("Dose máxima", f"{np.max(dose_map):.3f} Gy")
            st.metric("Dose mínima", f"{np.min(dose_map):.3f} Gy")
            st.metric("Desvio padrão", f"{np.std(dose_map):.3f} Gy")

        # Histograma de dose
        st.subheader("4. Histograma de Dose")
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        ax_hist.hist(dose_map.flatten(), bins=100, color="steelblue", edgecolor="black")
        ax_hist.set_xlabel("Dose (Gy)")
        ax_hist.set_ylabel("Frequência (pixels)")
        ax_hist.set_title("Distribuição de dose no filme")
        ax_hist.axvline(np.mean(dose_map), color="red", linestyle="--", label=f"Média = {np.mean(dose_map):.3f} Gy")
        ax_hist.legend()
        st.pyplot(fig_hist)

        # Download do mapa de dose como TIFF
        dose_uint16 = (np.clip(dose_map / np.max(dose_map), 0, 1) * 65535).astype(np.uint16) if np.max(dose_map) > 0 else np.zeros_like(dose_map, dtype=np.uint16)
        tiff_bytes = io.BytesIO()
        import tifffile
        tifffile.imwrite(tiff_bytes, dose_uint16)
        st.download_button(
            "⬇️ Baixar mapa de dose (TIFF 16-bit)",
            data=tiff_bytes.getvalue(),
            file_name="mapa_dose.tif",
            mime="image/tiff",
        )

        # Exportar CSV com valores por pixel (amostragem)
        st.subheader("5. Exportar dados brutos")
        h, w = dose_map.shape
        sample_step = max(1, min(h, w) // 200)  # Limitar para ~40k pontos
        ys, xs = np.mgrid[0:h:sample_step, 0:w:sample_step]
        export_df = pd.DataFrame({
            "y": ys.ravel(),
            "x": xs.ravel(),
            "NOD": nod_map[ys, xs].ravel(),
            "Dose_Gy": dose_map[ys, xs].ravel(),
        })
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Exportar dados amostrados (CSV)",
            data=csv_bytes,
            file_name="dados_dose.csv",
            mime="text/csv",
        )


# ============================================================
# ABA 3: GUIA RÁPIDO
# ============================================================
elif app_mode == "📖 Guia Rápido":
    st.title("📖 Guia Rápido — Filmes Radiocrômicos EBT3/EBT4")
    st.markdown(
        """
        ## Fundamentos

        **Filmes radiocrômicos** escurecem proporcionalmente à dose de radiação absorvida.
        O filme EBT3/EBT4 da Ashland é o padrão ouro para dosimetria independente em radioterapia.

        ### Canal Vermelho
        Após irradiado, o filme absorve mais luz no **espectro vermelho**.
        Por isso analisamos exclusivamente o canal R (ou BGR[2] no OpenCV).

        ### Net Optical Density (NOD)
        $$NOD = \\log_{10}\\left(\\frac{R_{bg}}{R_{irr}}\\right)$$

        Onde $R_{bg}$ é a reflectância do filme não irradiado e $R_{irr}$ do irradiado.

        ### Modelos de Calibração
        | Modelo | Fórmula | Quando usar |
        |--------|---------|-------------|
        | **Power Law** | $D = a \\cdot NOD^b$ | EBT3/EBT4 padrão, curva suave |
        | **Polinomial 3º** | $D = a + bNOD + cNOD² + dNOD³$ | Flexível, evitar extrapolação |
        | **Spline** | Interpolação cúbica | Muitos pontos, precisão máxima |
        | **Racional** | $(a+bNOD)/(1+cNOD)$ | Assíntota física |

        ### Workflow Recomendado
        1. Escaneie filmes em **modo de transmissão** ou **reflexão** com scanner plano de 16-bit
        2. Mantenha o tempo entre irradiacão e leitura constante (4-24h para EBT3, 2h para EBT4)
        3. Use a mesma imagem de controle (background) para calibração e análise
        4. Escolha **Power Law** se tiver 5-7 pontos; use **Spline** se tiver 10+ pontos
        5. Sempre verifique o $R²$ do ajuste

        ### Referências Implementadas
        - **Dosepy** (LuisOlivaresJ): leitura TIFF 16-bit, densidade óptica
        - **CHROMO** (matteobama): polinomial/spline
        - **radiochromic-film** (ckswilliams): reflectância, power law
        """
    )
