import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model
from scipy.stats import jarque_bera
from utils.data import get_data, TICKERS_DEFAULT

def show():
    st.header("Módulo 3 · Modelos ARCH/GARCH")
    st.caption("Modelado de la volatilidad condicional y pronóstico de riesgo.")

    # --- 1. Justificación ---
    with st.expander("¿Por qué un modelo de volatilidad condicional?", expanded=False):
        st.write("""
        Los modelos ARCH/GARCH capturan el 'agrupamiento de volatilidad', permitiendo que la varianza 
        dependa de errores pasados y volatilidades previas, mejorando la precisión del riesgo.
        """)

    # --- Configuración del modelo ---
    with st.sidebar:
        st.subheader("Configuración GARCH")
        ticker = st.selectbox("Activo representativo", options=list(TICKERS_DEFAULT.keys()), key="garch_ticker")
        
        # Opciones amigables para el usuario
        dist_opcion = st.selectbox("Distribución de errores", ["Normal", "t-Student", "Skewed t-Student"], index=1)
        
        # MAPEO CORRECTO PARA LA LIBRERÍA ARCH (Solución al ValueError)
        dist_map = {
            "Normal": "normal",
            "t-Student": "t",
            "Skewed t-Student": "skewt"
        }
        dist_final = dist_map[dist_opcion]
        
        horizonte_pro = st.slider("Pasos de pronóstico", 5, 30, 10)

    # --- Carga de datos ---
    df = get_data(ticker, period="5y")
    if df is None or df.empty: return

    # Escalamiento por 100 para evitar problemas de convergencia (estándar en GARCH)
    returns = 100 * np.log(df['Close'] / df['Close'].shift(1)).dropna()

    # --- 2. Ajuste de Especificaciones ---
    st.subheader(f"Modelado de Volatilidad para {ticker}")
    
    # Especificaciones requeridas: ARCH(1), GARCH(1,1) y una adicional (EGARCH)
    modelos_spec = {
        "ARCH(1)": {"p": 1, "q": 0, "vol": "GARCH"},
        "GARCH(1,1)": {"p": 1, "q": 1, "vol": "GARCH"},
        "EGARCH(1,1)": {"p": 1, "q": 1, "vol": "EGARCH"}
    }

    res_dict = {}
    with st.spinner("Ajustando modelos..."):
        for nombre, params in modelos_spec.items():
            try:
                mod = arch_model(returns, p=params["p"], q=params["q"], vol=params["vol"], dist=dist_final)
                res_dict[nombre] = mod.fit(disp="off")
            except Exception as e:
                st.warning(f"No se pudo ajustar {nombre}: {e}")

    # --- 3. Tabla Comparativa ---
    if res_dict:
        st.markdown("### Comparativa de Modelos")
        metrics = pd.DataFrame({
            "Log-Likelihood": [res.loglikelihood for res in res_dict.values()],
            "AIC": [res.aic for res in res_dict.values()],
            "BIC": [res.bic for res in res_dict.values()]
        }, index=res_dict.keys())
        st.table(metrics.style.highlight_min(subset=['AIC', 'BIC'], color='#004d00'))

        # Usamos GARCH(1,1) para diagnóstico y pronóstico
        if "GARCH(1,1)" in res_dict:
            res_final = res_dict["GARCH(1,1)"]
            
            # --- 4. Diagnóstico ---
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                # Gráfico de residuos estandarizados
                std_resid = res_final.resid / res_final.conditional_volatility
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(y=std_resid, mode='lines', name='Residuos std'))
                fig_res.update_layout(title="Residuos Estandarizados", height=300, template="plotly_dark")
                st.plotly_chart(fig_res, use_container_width=True)
            
            with c2:
                # Prueba Jarque-Bera sobre residuos
                jb_stat, jb_p = jarque_bera(std_resid.dropna())
                st.metric("Jarque-Bera (Residuos)", f"{jb_stat:.2f}")
                if jb_p < 0.05:
                    st.error(f"p-valor: {jb_p:.4f} (No Normalidad)")
                else:
                    st.success(f"p-valor: {jb_p:.4f} (Normalidad)")

            # --- 5. Pronóstico ---
            st.divider()
            st.subheader(f"Pronóstico de Volatilidad a {horizonte_pro} días")
            forecasts = res_final.forecast(horizon=horizonte_pro, reindex=False)
            vol_forecast = np.sqrt(forecasts.variance.values[-1])
            
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=list(range(1, horizonte_pro+1)), y=vol_forecast, mode='lines+markers'))
            fig_f.update_layout(xaxis_title="Días", yaxis_title="Volatilidad Pronosticada (%)", template="plotly_dark")
            st.plotly_chart(fig_f, use_container_width=True)