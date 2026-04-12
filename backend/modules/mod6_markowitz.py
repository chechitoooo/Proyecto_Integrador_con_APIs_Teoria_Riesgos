import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data import get_data, TICKERS_DEFAULT

def show():
    st.header("Módulo 6 · Optimización de Markowitz")
    st.caption("Construcción de la frontera eficiente y selección de portafolios óptimos.")

    # --- 1. CARGA DE DATOS ---
    tickers = list(TICKERS_DEFAULT.keys())
    
    with st.spinner("Descargando datos de todos los activos..."):
        try:
            # Creamos un dataframe con los precios de cierre de todos los tickers
            all_prices = {}
            for t in tickers:
                df_temp = get_data(t, period="2y")
                if df_temp is not None:
                    all_prices[t] = df_temp['Close']
            
            df_prices = pd.DataFrame(all_prices).dropna()
            # Rendimientos logarítmicos
            returns = np.log(df_prices / df_prices.shift(1)).dropna()
        except Exception as e:
            st.error(f"Error al procesar datos: {e}")
            return

    # --- 2. MATRIZ DE CORRELACIÓN (Requisito 1) ---
    st.subheader("Matriz de Correlación")
    corr_matrix = returns.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- 3. SIMULACIÓN DE MONTECARLO (Requisito 2: 10,000 portafolios) ---
    num_portafolios = 10000
    n_assets = len(tickers)
    
    # Pre-asignar arrays para velocidad
    results = np.zeros((3, num_portafolios))
    weights_record = []

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    for i in range(num_portafolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights) # Normalizar para que sumen 1
        weights_record.append(weights)
        
        # Retorno y Riesgo anualizado
        p_ret = np.sum(weights * mean_returns)
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0,i] = p_std
        results[1,i] = p_ret
        results[2,i] = p_ret / p_std # Sharpe Ratio (Rf=0)

    df_sim = pd.DataFrame(results.T, columns=['Volatilidad', 'Retorno', 'Sharpe'])

    # --- 4. IDENTIFICACIÓN DE ÓPTIMOS (Requisito 4) ---
    max_sharpe_idx = df_sim['Sharpe'].idxmax()
    min_vol_idx = df_sim['Volatilidad'].idxmin()

    # --- 5. GRÁFICO FRONTERA EFICIENTE (Requisito 3) ---
    st.subheader("Frontera Eficiente")
    fig_fe = px.scatter(df_sim, x='Volatilidad', y='Retorno', color='Sharpe',
                        labels={'Retorno': 'Retorno Esperado', 'Volatilidad': 'Riesgo (Desv. Est.)'})
    
    # Añadir puntos estrella para los óptimos
    fig_fe.add_trace(go.Scatter(x=[df_sim.iloc[max_sharpe_idx]['Volatilidad']], 
                                y=[df_sim.iloc[max_sharpe_idx]['Retorno']],
                                mode='markers', name='Máx. Sharpe', 
                                marker=dict(color='yellow', size=15, symbol='star')))
    
    fig_fe.add_trace(go.Scatter(x=[df_sim.iloc[min_vol_idx]['Volatilidad']], 
                                y=[df_sim.iloc[min_vol_idx]['Retorno']],
                                mode='markers', name='Mín. Varianza', 
                                marker=dict(color='white', size=15, symbol='diamond')))
    
    st.plotly_chart(fig_fe, use_container_width=True)

    # --- 6. TABLA DE COMPOSICIÓN (Requisito 5) ---
    st.subheader("Composición de Portafolios Óptimos")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**Cartera Máximo Sharpe**")
        comp_sharpe = pd.DataFrame({'Activo': tickers, 'Peso': weights_record[max_sharpe_idx]})
        st.dataframe(comp_sharpe.set_index('Activo').style.format("{:.2%}"))
        
    with c2:
        st.write("**Cartera Mínima Varianza**")
        comp_min = pd.DataFrame({'Activo': tickers, 'Peso': weights_record[min_vol_idx]})
        st.dataframe(comp_min.set_index('Activo').style.format("{:.2%}"))