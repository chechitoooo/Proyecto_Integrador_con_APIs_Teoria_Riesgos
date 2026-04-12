import streamlit as st
import pandas as pd
import numpy as np
from utils.data import get_data, TICKERS_DEFAULT

def show():
    st.header("Módulo 7 · Sistema de Señales ★")
    st.markdown("---")

    # --- CONFIGURACIÓN DE UMBRALES ---
    with st.sidebar:
        st.subheader("Configuración de Alertas")
        ticker = st.selectbox("Seleccionar Activo", options=list(TICKERS_DEFAULT.keys()), key="sig_ticker")
        rsi_up = st.slider("Umbral Sobrecompra (RSI)", 60, 80, 70)
        rsi_down = st.slider("Umbral Sobreventa (RSI)", 20, 40, 30)
        st.info("💡 Los umbrales afectan el color de las tarjetas en tiempo real.")

    # --- OBTENCIÓN DE DATOS ---
    df = get_data(ticker, period="1y")
    if df is None: return

    # --- CÁLCULOS TÉCNICOS NATIVOS (Sin pandas-ta) ---
    # 1. Medias Móviles
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # 2. RSI Nativo
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 3. MACD Nativo
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. Bandas de Bollinger Nativas
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BBU'] = df['MA20'] + (df['STD20'] * 2)
    df['BBL'] = df['MA20'] - (df['STD20'] * 2)

    # Datos actuales y anteriores
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # --- LÓGICA DE GENERACIÓN DE SEÑALES ---
    signals = []

    # 1. MACD (Cruce)
    if curr['MACD'] > curr['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']:
        signals.append({"name": "MACD", "status": "COMPRA", "desc": "Cruce alcista", "color": "green"})
    elif curr['MACD'] < curr['Signal_Line'] and prev['MACD'] >= prev['Signal_Line']:
        signals.append({"name": "MACD", "status": "VENTA", "desc": "Cruce bajista", "color": "red"})
    else:
        signals.append({"name": "MACD", "status": "NEUTRAL", "desc": "Sin cruces", "color": "blue"})

    # 2. RSI
    if curr['RSI_14'] >= rsi_up:
        signals.append({"name": "RSI", "status": "SOBRECOMPRA", "desc": f"Nivel: {curr['RSI_14']:.2f}", "color": "red"})
    elif curr['RSI_14'] <= rsi_down:
        signals.append({"name": "RSI", "status": "SOBREVENTA", "desc": f"Nivel: {curr['RSI_14']:.2f}", "color": "green"})
    else:
        signals.append({"name": "RSI", "status": "NEUTRAL", "desc": f"Estable: {curr['RSI_14']:.2f}", "color": "blue"})

    # 3. Bollinger
    if curr['Close'] >= curr['BBU']:
        signals.append({"name": "Bollinger", "status": "VENTA", "desc": "Precio sobre banda superior", "color": "red"})
    elif curr['Close'] <= curr['BBL']:
        signals.append({"name": "Bollinger", "status": "COMPRA", "desc": "Precio bajo banda inferior", "color": "green"})
    else:
        signals.append({"name": "Bollinger", "status": "NEUTRAL", "desc": "Dentro del canal", "color": "blue"})

    # 4. Medias Móviles
    if curr['SMA50'] > curr['SMA200'] and prev['SMA50'] <= prev['SMA200']:
        signals.append({"name": "Medias", "status": "GOLDEN CROSS", "desc": "Tendencia alcista fuerte", "color": "green"})
    elif curr['SMA50'] < curr['SMA200'] and prev['SMA50'] >= prev['SMA200']:
        signals.append({"name": "Medias", "status": "DEATH CROSS", "desc": "Tendencia bajista fuerte", "color": "red"})
    else:
        status = "ALCISTA" if curr['SMA50'] > curr['SMA200'] else "BAJISTA"
        signals.append({"name": "Medias", "status": status, "desc": "Sin cruces nuevos", "color": "blue"})

    # --- RENDERIZADO DEL PANEL ---
    st.subheader(f"Panel de Decisiones: {ticker}")
    cols = st.columns(len(signals))
    
    for i, sig in enumerate(signals):
        with cols[i]:
            with st.container(border=True):
                if sig['color'] == "green": st.success(f"**{sig['name']}**")
                elif sig['color'] == "red": st.error(f"**{sig['name']}**")
                else: st.info(f"**{sig['name']}**")
                st.write(f"### {sig['status']}")
                st.caption(sig['desc'])

    st.markdown("---")
    st.warning("⚠️ **Aviso:** Estas señales son automáticas. No constituyen recomendación de inversión.")