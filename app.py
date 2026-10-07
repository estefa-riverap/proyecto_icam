import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Estimador ICAM - INVEMAR",
    page_icon="🌊",
    layout="centered",
)

# ----------------------------------------------------------------------
# Carga del bundle
# ----------------------------------------------------------------------
@st.cache_resource
def cargar_bundle():
    return joblib.load("modelo_rf_icam.pkl")

bundle = joblib.load("modelo_rf_icam.pkl")
modelo = bundle["modelo"]
FEATURES = bundle["features"]

# Test de humo: verificar contrato modelo ↔ bundle
if list(modelo.feature_names_in_) != list(FEATURES):
    st.error(
        "⚠️ El bundle está inconsistente: las features del modelo no coinciden "
        "con las del bundle. Revisa cómo se generó el .pkl."
    )
    st.stop()

# ----------------------------------------------------------------------
# Categorización ICAM (escala INVEMAR 0-100)
# ----------------------------------------------------------------------
def categorizar_icam(icam: float):
    if icam < 25:   return "Pésima",     "#B71C1C"
    if icam < 50:   return "Inadecuada", "#E65100"
    if icam < 70:   return "Aceptable",  "#F9A825"
    if icam < 90:   return "Adecuada",   "#2E7D32"
    return "Óptima", "#1B5E20"

# ----------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------
st.title("🌊 Estimador del ICAM")
st.caption(
    "Estimación del Índice de Calidad de Aguas Marinas y Costeras (ICAM) "
    "mediante Random Forest — datos históricos INVEMAR"
)

metricas = bundle.get("metricas", {})
if metricas.get("r2_test") is not None:
    st.info(
        f"R² del modelo sobre el conjunto de prueba: **{metricas['r2_test']:.3f}** "
        f"· RMSE: {metricas['rmse_test']:.2f} · MAE: {metricas['mae_test']:.2f}"
    )

st.markdown("---")

# ----------------------------------------------------------------------
# Formulario
# ----------------------------------------------------------------------
with st.form("formulario_icam"):
    st.subheader("Parámetros fisicoquímicos y microbiológicos")

    col1, col2 = st.columns(2)
    with col1:
        cte = st.number_input(
            f"Coliformes termotolerantes ({bundle['unidades']['cte']})",
            min_value=0.0, value=140.0, step=1.0, format="%.2f",
        )
        no3 = st.number_input(
            f"Nitratos ({bundle['unidades']['no3']})",
            min_value=0.0, value=19.2, step=0.1, format="%.3f",
        )
    with col2:
        po4 = st.number_input(
            f"Ortofosfatos ({bundle['unidades']['po4']})",
            min_value=0.0, value=6.615, step=0.1, format="%.3f",
        )
        sst = st.number_input(
            f"Sólidos suspendidos totales ({bundle['unidades']['sst']})",
            min_value=0.0, value=18.6, step=0.1, format="%.2f",
        )

    st.subheader("Contexto geográfico")
    col3, col4 = st.columns(2)
    with col3:
        sustrato = st.selectbox(
            "Sustrato",
            options=bundle["variables_categoricas"]["sustrato"],
        )
    with col4:
        region = st.selectbox(
            "Región",
            options=bundle["variables_categoricas"]["region"],
        )

    enviado = st.form_submit_button("Estimar ICAM", use_container_width=True)

# ----------------------------------------------------------------------
# Predicción
# ----------------------------------------------------------------------
if enviado:
    # 1. Variables numéricas crudas (SIN log)
    num = {
        "cte": cte,
        "po4": po4,
        "no3": no3,
        "sst": sst,
    }

    # 2. Construcción de las dummies de contexto
    ref_sustrato = bundle["categorias_referencia"]["sustrato"]
    ref_region   = bundle["categorias_referencia"]["region"]

    dummies = {
        "sustrato_Agua Marina":      int(sustrato == "Agua Marina"),
        "region_CARIBE":             int(region == "CARIBE"),
        "region_CARIBE INSULAR":     int(region == "CARIBE INSULAR"),
        "region_PACIFICO":           int(region == "PACIFICO"),
    }

    # 3. Ensamblar el vector de entrada en el ORDEN del modelo
    fila = {**num, **dummies}
    X_input = pd.DataFrame([[fila[f] for f in FEATURES]], columns=FEATURES)

    # 4. Predicción
    icam = float(modelo.predict(X_input)[0])
    icam = max(0.0, min(100.0, icam))
    categoria, color = categorizar_icam(icam)

    st.markdown("---")
    st.subheader("Resultado")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("ICAM estimado", f"{icam:.2f} / 100")
    with c2:
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:18px;
                        border-radius:10px; text-align:center;
                        color:white; font-size:22px; font-weight:bold;">
                {categoria}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.progress(icam / 100.0)

    with st.expander("🔍 Ver detalle del cálculo"):
        st.write("**Entradas numéricas (crudas):**")
        st.write(pd.DataFrame([num]))
        st.write("**Entradas categóricas codificadas:**")
        st.write(pd.DataFrame([dummies]))
        st.write("**Vector final enviado al modelo:**")
        st.write(X_input)

    st.warning(
        "⚠️ Estimación basada en un modelo de Machine Learning entrenado con "
        "datos históricos del INVEMAR. No reemplaza el cálculo oficial del "
        "ICAM ni el criterio técnico de la autoridad ambiental."
    )

st.markdown("---")
st.caption(
    "Modelo: Random Forest · Entradas: 4 fisicoquímicas + sustrato + región"
)