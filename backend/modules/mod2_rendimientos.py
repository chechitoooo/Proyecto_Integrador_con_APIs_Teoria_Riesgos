import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import jarque_bera, shapiro, norm
import statsmodels.api as sm

from utils.data import get_data, TICKERS_DEFAULT

def show():
    st.header("Módulo 2 · Rendimientos")
    st.caption("Caracterización estadística de los rendimientos y análisis de normalidad.")

    # ── Sidebar: Configuración ──
    with st.sidebar:
        st.subheader("Configuración Mód. 2")
        ticker_input = st.text_input("Ticker personalizado", key="m2_ticker").upper().strip()
        ticker_sugerido = st.selectbox("O elige uno sugerido", options=list(TICKERS_DEFAULT.keys()), key="m2_select")
        ticker = ticker_input if ticker_input else ticker_sugerido

        periodo = st.selectbox("Horizonte histórico", options=["1y", "2y", "5y", "max"], index=1)
        tipo_rend = st.radio("Tipo de rendimiento", options=["Simple", "Logarítmico"], horizontal=True)

    # ── Carga de Datos ──
    try:
        with st.spinner(f"Analizando {ticker}..."):
            df = get_data(ticker, period=periodo)
            if df is None or df.empty:
                st.error("No hay datos disponibles.")
                return
    except Exception as e:
        st.error(f"Error: {e}")
        return

    # ── Cálculo de Rendimientos (Requisito 1) ──
    df["Rend_Simple"] = df["Close"].pct_change()
    df["Rend_Log"] = np.log(df["Close"] / df["Close"].shift(1))
    
    col_rend = "Rend_Simple" if tipo_rend == "Simple" else "Rend_Log"
    rend = df[col_rend].dropna()

    # ── Estadísticas Descriptivas (Requisito 2) ──
    st.subheader(f"Estadísticas Descriptivas ({tipo_rend})")
    media = rend.mean()
    desv = rend.std()
    asim = rend.skew()
    curt = rend.kurtosis() # Exceso de curtosis (Normal = 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Media", f"{media:.5f}")
    m2.metric("Desv. Estándar", f"{desv:.5f}")
    m3.metric("Asimetría (Skew)", f"{asim:.3f}")
    m4.metric("Curtosis (Exceso)", f"{curt:.3f}")

    st.divider()

    # ── Gráficos Interactivos (Requisitos 3, 4 y 5) ──
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Histograma con Curva Normal", 
            "Gráfico Q-Q (Cuantiles)",
            "Boxplot de Rendimientos", 
            "Serie Temporal (Volatility Clustering)"
        ),
        vertical_spacing=0.15
    )

    # 1. Histograma con PDF Normal (Requisito 3)
    fig.add_trace(go.Histogram(x=rend, histnorm='probability density', name="Rendimientos", marker_color='#3366CC', opacity=0.6), row=1, col=1)
    x_range = np.linspace(rend.min(), rend.max(), 100)
    y_normal = norm.pdf(x_range, media, desv)
    fig.add_trace(go.Scatter(x=x_range, y=y_normal, mode='lines', name='Normal Teórica', line=dict(color='red', width=2)), row=1, col=1)

    # 2. Gráfico Q-Q (Requisito 4)
    qq = sm.ProbPlot(rend, dist=norm, fit=True)
    fig.add_trace(go.Scatter(x=qq.theoretical_quantiles, y=np.sort(rend), mode='markers', name='Q-Q', marker=dict(size=4, color='#3366CC')), row=1, col=2)
    line_range = [min(qq.theoretical_quantiles), max(qq.theoretical_quantiles)]
    fig.add_trace(go.Scatter(x=line_range, y=line_range, mode='lines', name='45° Line', line=dict(color='red', dash='dash')), row=1, col=2)

    # 3. Boxplot (Requisito 5)
    fig.add_trace(go.Box(y=rend, name="Rendimientos", boxpoints='outliers', marker_color='#3366CC'), row=2, col=1)

    # 4. Serie temporal (Para discutir Agrupamiento de Volatilidad)
    fig.add_trace(go.Scatter(x=rend.index, y=rend, mode='lines', name='Retornos', line=dict(width=1)), row=2, col=2)

    fig.update_layout(height=800, showlegend=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # ── Pruebas de Normalidad e Interpretación (Requisito 6 y 7) ──
    st.divider()
    st.subheader("Pruebas de Normalidad")
    
    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(rend)
    # Shapiro-Wilk (Muestreo si N > 5000)
    rend_sample = rend.sample(min(len(rend), 5000))
    sh_stat, sh_p = shapiro(rend_sample)

    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Jarque-Bera p-valor:** {jb_p:.6f}")
        if jb_p < 0.05: st.error("Rechazamos H0: No es Normal")
        else: st.success("No rechazamos H0: Es Normal")

    with c2:
        st.info(f"**Shapiro-Wilk p-valor:** {sh_p:.6f}")
        if sh_p < 0.05: st.error("Rechazamos H0: No es Normal")
        else: st.success("No rechazamos H0: Es Normal")

    # ── Discusión de Hechos Estilizados (Requisito 8) ──
    st.subheader("Discusión de Hechos Estilizados")
    with st.expander("Haz clic para ver el análisis de la distribución"):
        st.markdown(f"""
        * **Colas Pesadas (Leptocurtosis):** Con una curtosis de **{curt:.2f}**, el activo presenta colas más anchas que la normal. En el Q-Q Plot, esto se observa en las desviaciones de los extremos.
        * **Agrupamiento de Volatilidad:** En el gráfico de la serie temporal, se observa que los rendimientos extremos tienden a aparecer en bloques (clusters), rompiendo la independencia temporal.
        * **Efecto Apalancamiento:** La asimetría de **{asim:.2f}** sugiere si existe una respuesta desproporcionada de la volatilidad ante choques negativos.
        """)

    # ── Tabla de Datos ──
    with st.expander("Ver Datos Tabulados"):
        st.dataframe(df[[col_rend]].tail(10).style.format("{:.6f}"))

if __name__ == "__main__":
    show()