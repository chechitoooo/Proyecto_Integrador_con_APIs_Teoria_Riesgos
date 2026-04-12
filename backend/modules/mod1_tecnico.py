import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Importación de tus utilidades (Asegúrate de que existan en tu proyecto)
from utils.data import get_data, calcular_indicadores, TICKERS_DEFAULT

def show():
    # --- ENCABEZADO ---
    st.header("Módulo 1: Análisis Técnico")
    st.markdown("""
    Permite explorar el comportamiento histórico de cada activo y visualizar indicadores técnicos avanzados.
    """)

    # --- SIDEBAR: CONTROLES ---
    with st.sidebar:
        st.subheader("🔍 Selección de Activo")
        
        # 1. Selector dinámico (Punto 1 de la consigna)
        ticker_input = st.text_input(
            "Ticker personalizado", 
            placeholder="Ej: AAPL, BTC-USD, MSFT",
            help="Ingresa el ticker oficial de Yahoo Finance"
        ).upper().strip()

        ticker_sugerido = st.selectbox(
            "O elige un activo sugerido",
            options=list(TICKERS_DEFAULT.keys()),
            format_func=lambda t: f"{t} — {TICKERS_DEFAULT[t]}"
        )

        ticker = ticker_input if ticker_input else ticker_sugerido

        # Configuración de visualización
        with st.expander("📅 Rango y Estilo", expanded=True):
            periodo = st.selectbox("Horizonte inicial", ["1y", "2y", "5y", "max"], index=0)
            tipo_grafico = st.radio("Tipo de gráfico", ["Velas japonesas", "Línea"], horizontal=True)

        # 2. Parámetros ajustables (Punto 3, 4 y 7 de la consigna)
        with st.expander("⚙️ Parámetros de Indicadores"):
            sma_c = st.slider("SMA Corto", 5, 50, 20)
            sma_l = st.slider("SMA Largo", 20, 200, 50)
            rsi_p = st.slider("Período RSI", 5, 30, 14)
            stoch_p = st.slider("Período Estocástico", 5, 30, 14)
            bb_p = st.slider("Bollinger (N)", 5, 50, 20)
            bb_std = st.slider("Bollinger (Std)", 1.0, 3.0, 2.0, 0.5)

        # 3. Interruptores de visibilidad
        st.subheader("📈 Indicadores Visibles")
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            show_sma = st.checkbox("SMA", True)
            show_bb = st.checkbox("Bollinger", True)
            show_rsi = st.checkbox("RSI", True)
        with col_check2:
            show_ema = st.checkbox("EMA", True)
            show_macd = st.checkbox("MACD", True)
            show_stoch = st.checkbox("Estocástico", True)

    # --- CARGA DE DATOS ---
    try:
        with st.spinner(f"Descargando datos de {ticker}..."):
            df_raw = get_data(ticker, period=periodo)
            
            if df_raw is None or df_raw.empty:
                st.error("No se encontraron datos. Revisa el ticker.")
                return

            # Cálculo con los parámetros del sidebar
            df = calcular_indicadores(
                df_raw, 
                sma_periodos=[sma_c, sma_l],
                rsi_periodo=rsi_p,
                bb_periodo=bb_p,
                bb_std=bb_std
            )
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return

    # --- FILTRO DE FECHAS (Punto 2 de la consigna) ---
    st.subheader(f"Análisis de {ticker}")
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        f_inicio = st.date_input("Desde", df.index.min(), min_value=df.index.min(), max_value=df.index.max())
    with c_f2:
        f_fin = st.date_input("Hasta", df.index.max(), min_value=df.index.min(), max_value=df.index.max())

    df_view = df.loc[str(f_inicio):str(f_fin)].copy()

    # --- MÉTRICAS DE RESUMEN ---
    m1, m2, m3, m4 = st.columns(4)
    last_p = df_view['Close'].iloc[-1]
    change = (last_p / df_view['Close'].iloc[0] - 1) * 100
    m1.metric("Precio Actual", f"${last_p:,.2f}")
    m2.metric("Retorno Período", f"{change:.2f}%", delta=f"{change:.2f}%")
    m3.metric("RSI", f"{df_view['RSI'].iloc[-1]:.1f}")
    m4.metric("Volatilidad", f"{(df_view['Close'].pct_change().std() * 100):.2f}%")

    # --- CONSTRUCCIÓN DEL GRÁFICO MULTI-PANEL ---
    # Determinamos cuántos subplots necesitamos
    extra_rows = sum([show_rsi, show_macd, show_stoch])
    rows = 1 + extra_rows
    row_heights = [0.5] + ([0.15] * extra_rows)
    
    fig = make_subplots(
        rows=rows, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=row_heights
    )

    # PANEL 1: PRECIO e INDICADORES DE SUPERPOSICIÓN
    if tipo_grafico == "Velas japonesas":
        fig.add_trace(go.Candlestick(
            x=df_view.index, open=df_view['Open'], high=df_view['High'],
            low=df_view['Low'], close=df_view['Close'], name="Precio"
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Close'], name="Precio", line=dict(color='#1f77b4')), row=1, col=1)

    if show_sma:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view[f'SMA_{sma_c}'], name=f'SMA {sma_c}', line=dict(width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view[f'SMA_{sma_l}'], name=f'SMA {sma_l}', line=dict(width=1.5)), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['BB_upper'], name="BB Superior", line=dict(dash='dot', color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['BB_lower'], name="BB Inferior", line=dict(dash='dot', color='gray'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    # PANELES SECUNDARIOS (RSI, MACD, STOCH)
    current_r = 2
    if show_rsi:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['RSI'], name="RSI", line=dict(color='#8e44ad')), row=current_r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_r, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_r, col=1)
        current_r += 1

    if show_macd:
        fig.add_trace(go.Bar(x=df_view.index, y=df_view['MACD_Hist'], name="Histograma"), row=current_r, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MACD'], name="MACD"), row=current_r, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['MACD_Signal'], name="Señal"), row=current_r, col=1)
        current_r += 1

    if show_stoch:
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['%K'], name="%K"), row=current_r, col=1)
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['%D'], name="%D"), row=current_r, col=1)
        current_r += 1

    # Estética Pro
    fig.update_layout(
        height=400 + (150 * extra_rows),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(t=20, b=20, l=0, r=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- PANEL EXPLICATIVO (Punto 8 de la consigna) ---
    st.divider()
    st.subheader("📘 Guía de Indicadores")
    cols = st.columns(3)
    with cols[0]:
        with st.expander("¿Qué es la SMA?"):
            st.write("Promedio simple de precios. Los cruces de la SMA corta sobre la larga indican cambios de tendencia.")
    with cols[1]:
        with st.expander("¿Qué indica el RSI?"):
            st.write("Mide la velocidad del precio. >70 sugiere sobrecompra (caro), <30 sobreventa (barato).")
    with cols[2]:
        with st.expander("Bandas de Bollinger"):
            st.write("Miden la volatilidad. El precio suele mantenerse dentro de las bandas el 95% del tiempo.")

    # --- TABLA DE DATOS ---
    with st.expander("Visualizar datos tabulados"):
        st.dataframe(df_view.tail(30).sort_index(ascending=False), use_container_width=True)

# Ejecución
if __name__ == "__main__":
    show()
 