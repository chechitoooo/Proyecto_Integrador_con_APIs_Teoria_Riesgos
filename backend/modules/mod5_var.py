import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.stats import norm
from utils.data import get_data, TICKERS_DEFAULT

def show():
    st.header("Mód. 5: VaR y CVaR")
    st.caption("Cuantificación de la pérdida potencial máxima bajo diferentes metodologías.")

    # --- Configuración en Sidebar ---
    with st.sidebar:
        st.subheader("Parámetros de Riesgo")
        ticker = st.selectbox("Seleccionar Activo", options=list(TICKERS_DEFAULT.keys()), key="var_ticker")
        confianza = st.select_slider("Nivel de Confianza", options=[0.95, 0.99], value=0.95)
        inversion = st.number_input("Inversión a Riesgo (USD)", value=10000, step=1000)

    # --- Carga de Datos ---
    df = get_data(ticker, period="2y")
    if df is None: return
    returns = df['Close'].pct_change().dropna()

    # --- 1. VaR Paramétrico (Gausiano) ---
    mu = returns.mean()
    sigma = returns.std()
    var_param_diario = norm.ppf(1 - confianza, mu, sigma)
    var_param_anual = var_param_diario * np.sqrt(252)

    # --- 2. VaR Simulación Histórica ---
    var_hist_diario = np.percentile(returns, (1 - confianza) * 100)

    # --- 3. VaR Montecarlo (Mínimo 10,000 simulaciones) ---
    n_sims = 10000
    sim_returns = np.random.normal(mu, sigma, n_sims)
    var_mc_diario = np.percentile(sim_returns, (1 - confianza) * 100)

    # --- 4. Expected Shortfall (CVaR) ---
    cvar_diario = returns[returns <= var_hist_diario].mean()

    # --- Visualización ---
    st.subheader(f"Distribución de Rendimientos y Líneas de Riesgo ({ticker})")
    
    fig = ff.create_distplot([returns], ['Rendimientos'], bin_size=0.005, show_rug=False)
    
    # Agregar líneas de VaR y CVaR
    fig.add_vline(x=var_hist_diario, line_dash="dash", line_color="red", 
                  annotation_text=f"VaR Hist ({confianza:.0%})")
    fig.add_vline(x=cvar_diario, line_dash="dot", line_color="orange", 
                  annotation_text="CVaR (Pérdida Promedio Extrema)")
    
    fig.update_layout(template="plotly_dark", showlegend=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. Tabla Comparativa de Métodos ---
    st.subheader("Tabla Comparativa de Métodos")
    
    data_metrics = {
        "Método": ["Paramétrico (Diario)", "Paramétrico (Anualizado)", "Simulación Histórica", "Montecarlo (10k sims)", "Expected Shortfall (CVaR)"],
        "Valor (%)": [
            f"{var_param_diario:.2%}", f"{var_param_anual:.2%}", 
            f"{var_hist_diario:.2%}", f"{var_mc_diario:.2%}", f"{cvar_diario:.2%}"
        ],
        "Pérdida en USD": [
            f"${abs(var_param_diario * inversion):,.2f}", f"${abs(var_param_anual * inversion):,.2f}",
            f"${abs(var_hist_diario * inversion):,.2f}", f"${abs(var_mc_diario * inversion):,.2f}",
            f"${abs(cvar_diario * inversion):,.2f}"
        ]
    }
    
    st.table(pd.DataFrame(data_metrics))

    # --- Interpretación ---
    st.info(f"**Interpretación:** Con un {confianza:.0%} de confianza, la pérdida máxima esperada en un día para una inversión de ${inversion:,.0f} es de aproximadamente {abs(var_hist_diario * inversion):,.2f} USD (según el método histórico).")