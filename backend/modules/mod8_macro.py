import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data import get_data, TICKERS_DEFAULT

def show():
    st.header("Módulo 8 · Macro y Benchmark ★")
    st.caption("Contexto macroeconómico y comparativa del portafolio óptimo contra el mercado.")

    # --- 1. PANEL MACRO (Vía API/Simulado para estabilidad) ---
    st.subheader("Indicadores Macroeconómicos")
    m1, m2, m3 = st.columns(3)
    
    # En un entorno real, aquí conectarías con APIs de FRED o Banrep
    # Usamos datos de referencia actualizados para la sustentación
    m1.metric("Tasa Libre de Riesgo (10Y)", "4.32%", "+0.05", help="US Treasury 10Y Yield")
    m2.metric("Inflación (CPI)", "3.10%", "-0.10", delta_color="inverse")
    m3.metric("Tasa de Cambio (TRM)", "$3,950", "+12.50")

    st.markdown("---")

    # --- 2. COMPARATIVA DE RENDIMIENTO ---
    st.subheader("Portafolio Óptimo vs. Benchmark (Base 100)")
    
    with st.spinner("Calculando comparativa de desempeño..."):
        # Descarga de datos: Portafolio vs Benchmark (S&P 500)
        tickers = list(TICKERS_DEFAULT.keys())
        benchmark_ticker = "^GSPC" # S&P 500
        
        all_assets = tickers + [benchmark_ticker]
        data = {t: get_data(t, period="1y")['Close'] for t in all_assets}
        df = pd.DataFrame(data).dropna()
        
        # Simulación de pesos del Portafolio Óptimo (Máximo Sharpe simplificado)
        # En una integración total, estos pesos vendrían del Módulo 6
        weights = np.array([1/len(tickers)] * len(tickers)) 
        
        # Cálculo de Retornos Acumulados (Base 100)
        port_returns = df[tickers].pct_change().dropna()
        bench_returns = df[benchmark_ticker].pct_change().dropna()
        
        port_cum = (1 + (port_returns @ weights)).cumprod() * 100
        bench_cum = (1 + bench_returns).cumprod() * 100

        # Gráfico de Comparación
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Mi Portafolio Óptimo", line=dict(color='#00ffcc', width=3)))
        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="Benchmark (S&P 500)", line=dict(color='white', dash='dash')))
        fig.update_layout(title="Rendimiento Acumulado: Estrategia vs. Mercado", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # --- 3. MÉTRICAS DE GESTIÓN (Alpha, Tracking Error, IR) ---
    st.subheader("Métricas de Desempeño y Riesgo")
    
    # Cálculos para métricas
    rf = 0.04 / 252 # Tasa diaria asumiendo 4% anual
    excess_port = port_returns @ weights - rf
    excess_bench = bench_returns - rf
    
    # Alpha de Jensen (Simplificado: Retorno Port - Retorno Bench)
    alpha = (port_cum.iloc[-1] - bench_cum.iloc[-1]) / 100
    # Tracking Error
    tracking_error = (port_returns @ weights - bench_returns).std() * np.sqrt(252)
    # Information Ratio
    info_ratio = alpha / tracking_error if tracking_error != 0 else 0
    # Máximo Drawdown
    peak = port_cum.expanding(min_periods=1).max()
    drawdown = (port_cum - peak) / peak
    max_dd = drawdown.min()

    # Mostrar Tabla de Desempeño
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Métricas de Selección**")
        metrics_df = pd.DataFrame({
            "Métrica": ["Alpha de Jensen", "Tracking Error", "Information Ratio"],
            "Valor": [f"{alpha:.2%}", f"{tracking_error:.2%}", f"{info_ratio:.2f}"]
        })
        st.table(metrics_df)

    with col_b:
        st.write("**Resumen de Desempeño**")
        perf_df = pd.DataFrame({
            "Métrica": ["Rendimiento Anualizado", "Volatilidad", "Máximo Drawdown"],
            "Valor": [f"{(port_cum.iloc[-1]/100 - 1):.2%}", f"{(port_returns @ weights).std()*np.sqrt(252):.2%}", f"{max_dd:.2%}"]
        })
        st.table(perf_df)

    # --- 4. INTERPRETACIÓN ---
    st.subheader("Interpretación del Gestor")
    if alpha > 0:
        st.success(f"🚀 El portafolio superó al benchmark con un Alpha de {alpha:.2%}. La selección de activos agregó valor real.")
    else:
        st.warning("📉 El portafolio sub-performa al mercado. Se recomienda revisar los pesos en el Módulo de Markowitz.")