import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from utils.data import get_data, TICKERS_DEFAULT


def show():
    st.header("Módulo 4 · CAPM y Beta")
    
    # --- 1. TASA LIBRE DE RIESGO ---
    with st.spinner("Consultando tasa libre de riesgo actualizada..."):
        df_rf = get_data("^TNX", period="5d")

        if df_rf is not None and not df_rf.empty:
            df_rf.index = pd.to_datetime(df_rf.index).tz_localize(None)
            df_rf = df_rf.sort_index()

            rf_anual = df_rf['Close'].iloc[-1] / 100
            st.success(f"Tasa Libre de Riesgo ($R_f$): **{rf_anual:.2%}**")
        else:
            rf_anual = 0.04
            st.warning("No se pudo obtener ^TNX. Usando 4.00%")

    # --- SIDEBAR ---
    with st.sidebar:
        st.subheader("Configuración CAPM")
        ticker = st.selectbox("Seleccionar Activo", options=list(TICKERS_DEFAULT.keys()))
        benchmark = "^GSPC"

    # --- 2. CARGA DE DATOS ---
    df_asset = get_data(ticker, period="2y")
    df_bench = get_data(benchmark, period="2y")

    if df_asset is None or df_bench is None:
        st.error("Error descargando datos.")
        return

    # 🔥 LIMPIEZA FUERTE (CLAVE PARA EVITAR EL ERROR)
    def clean_df(df):
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        df = df[~df.index.duplicated()]
        return df

    df_asset = clean_df(df_asset)
    df_bench = clean_df(df_bench)

    # --- 3. RENDIMIENTOS ---
    asset_ret = df_asset['Close'].pct_change().dropna()
    bench_ret = df_bench['Close'].pct_change().dropna()

    # 🔥 USAR MERGE EN LUGAR DE CONCAT (más seguro)
    data = pd.merge(
        asset_ret,
        bench_ret,
        left_index=True,
        right_index=True,
        how='inner'
    ).dropna()

    data.columns = ['Asset', 'Market']

    if data.empty:
        st.error("No hay datos suficientes después de sincronizar.")
        return

    # --- 4. REGRESIÓN (BETA) ---
    X = sm.add_constant(data['Market'])
    model = sm.OLS(data['Asset'], X).fit()
    beta = model.params['Market']

    # --- 5. CAPM ---
    rm_anual = data['Market'].mean() * 252
    expected_return = rf_anual + beta * (rm_anual - rf_anual)

    # --- 6. VISUALIZACIÓN ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Línea Característica del Activo")

        fig = px.scatter(
            data,
            x="Market",
            y="Asset",
            opacity=0.4,
            labels={
                'Market': 'Rendimiento Mercado',
                'Asset': 'Rendimiento Activo'
            }
        )

        # Línea de regresión
        x_line = np.array([data['Market'].min(), data['Market'].max()])
        y_line = model.params['const'] + beta * x_line

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            name="Regresión (Beta)",
            line=dict(color='orange', width=3)
        ))

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Beta Estimado", f"{beta:.2f}")
        st.metric("Retorno Esperado (CAPM)", f"{expected_return:.2%}")

        # Clasificación
        if beta > 1.1:
            st.error("Clasificación: Agresivo")
        elif beta < 0.9:
            st.success("Clasificación: Defensivo")
        else:
            st.info("Clasificación: Neutro")

    # --- 7. ANÁLISIS ---
    st.divider()
    st.info(
        f"**Análisis de Riesgo:** El Beta es {beta:.2f}. "
        f"El activo es {'más' if beta > 1 else 'menos'} volátil que el mercado."
    )