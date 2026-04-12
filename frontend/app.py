"""
frontend/app.py  — Dashboard Streamlit · Tema claro y profesional
Ejecutar DESPUÉS de levantar el backend:
    cd backend && uvicorn main:app --reload --port 8000
    cd frontend && streamlit run app.py
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm2

# ─── CONFIGURACIÓN PÁGINA ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Financiero · Teoría de Riesgo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS TEMA CLARO PROFESIONAL ───────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #F8FAFC; }

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 13px !important; }
[data-testid="stMetricValue"] { color: #1E293B !important; font-size: 26px !important; font-weight: 700 !important; }

.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

.page-header {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    padding: 20px 28px;
    border-radius: 14px;
    margin-bottom: 24px;
}
.page-header h2 { color: white !important; margin: 0; font-size: 22px; }
.page-header p  { color: rgba(255,255,255,0.85) !important; margin: 4px 0 0; font-size: 13px; }

.badge-success { display:block; background:#DCFCE7; color:#166534; border:1px solid #BBF7D0; border-radius:8px; padding:10px 16px; font-weight:600; font-size:14px; }
.badge-error   { display:block; background:#FEE2E2; color:#991B1B; border:1px solid #FECACA; border-radius:8px; padding:10px 16px; font-weight:600; font-size:14px; }
.badge-info    { display:block; background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; border-radius:8px; padding:10px 16px; font-weight:600; font-size:14px; }
.badge-warning { display:block; background:#FFFBEB; color:#92400E; border:1px solid #FDE68A; border-radius:8px; padding:10px 16px; font-weight:600; font-size:14px; }

.signal-buy  { background:#DCFCE7; color:#166534; border:2px solid #86EFAC; border-radius:12px; padding:18px; text-align:center; font-size:20px; font-weight:700; }
.signal-sell { background:#FEE2E2; color:#991B1B; border:2px solid #FCA5A5; border-radius:12px; padding:18px; text-align:center; font-size:20px; font-weight:700; }
.signal-neut { background:#EFF6FF; color:#1D4ED8; border:2px solid #93C5FD; border-radius:12px; padding:18px; text-align:center; font-size:20px; font-weight:700; }

.sig-card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:18px 12px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.sig-label { font-size:11px; font-weight:600; color:#94A3B8; letter-spacing:1px; margin-bottom:6px; }
.sig-estado { font-weight:700; font-size:15px; color:#1E293B; margin-top:6px; }
.sig-desc   { font-size:12px; color:#64748B; margin-top:4px; }

h1, h2, h3 { color: #1E293B !important; }
p { color: #475569; }

.stButton>button { border-radius: 8px; font-weight: 600; }

div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTES ───────────────────────────────────────────────────────────────
PRIMARY = "#6366F1"
SUCCESS = "#22C55E"
DANGER  = "#EF4444"
WARNING = "#F59E0B"
INFO    = "#3B82F6"
MUTED   = "#94A3B8"

PLOT_TPL = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FAFC",
    font=dict(family="Inter, sans-serif", color="#1E293B"),
    margin=dict(t=40, b=30, l=10, r=10),
)

def page_header(title, subtitle=""):
    st.markdown(
        f'<div class="page-header"><h2>{title}</h2>'
        + (f'<p>{subtitle}</p>' if subtitle else '') +
        '</div>', unsafe_allow_html=True
    )

def badge_html(text, kind="info"):
    st.markdown(f'<div class="badge-{kind}">{text}</div>', unsafe_allow_html=True)

# ─── API ──────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
        if r.status_code == 200:
            return r.json()
        st.error(f"Error API ({r.status_code}): {r.json().get('detail', r.text)}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar al backend en `localhost:8000`.")
        return None

def api_get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_tickers():
    data = api_get("/api/utils/tickers")
    return data["tickers"] if data else {
        "AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Alphabet","AMZN":"Amazon",
        "TSLA":"Tesla","NVDA":"NVIDIA","JPM":"JPMorgan","BAC":"Bank of America",
        "GLD":"Gold ETF","BTC-USD":"Bitcoin"
    }

TICKERS = get_tickers()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Dashboard Financiero")
    st.markdown("---")
    opcion = st.radio("Módulo", [
        "🏠 Portada",
        "📈 Módulo 1 · Técnico",
        "📉 Módulo 2 · Rendimientos",
        "🌊 Módulo 3 · ARCH/GARCH",
        "⚖️ Módulo 4 · CAPM y Beta",
        "🛡️ Módulo 5 · VaR y CVaR",
        "🎯 Módulo 6 · Markowitz",
        "🚦 Módulo 7 · Señales ★",
        "🌍 Módulo 8 · Macro ★",
    ], label_visibility="collapsed")
    st.markdown("---")
    health = api_get("/api/utils/health")
    if health:
        badge_html("🟢 Backend conectado", "success")
    else:
        badge_html("🔴 Backend desconectado", "error")
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Teoría de Riesgo · Javier Sierra")


# ══════════════════════════════════════════════════════════════════════════════
#  PORTADA
# ══════════════════════════════════════════════════════════════════════════════
if opcion == "🏠 Portada":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#6366F1,#8B5CF6);border-radius:16px;
                padding:36px 40px;margin-bottom:28px;">
        <h1 style="color:white;margin:0;font-size:30px;">📊 Dashboard de Análisis Financiero</h1>
        <p style="color:rgba(255,255,255,0.85);margin-top:8px;font-size:15px;">
            Proyecto Final · Teoría de Riesgo
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**👥 Integrantes**")
        st.write("• Sergio David Huertas Ramírez")
        st.write("• Sergio Andrés Prieto Orjuela")
        st.markdown("---")
        st.markdown("**📚 Materia:** Teoría de Riesgo")
        st.markdown("**👨‍🏫 Profesor:** Javier Sierra")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🏗️ Arquitectura**")
        st.markdown("- **Frontend:** Streamlit")
        st.markdown("- **Backend:** FastAPI REST")
        st.markdown("- **Comunicación:** `requests` HTTP")
        st.markdown("- **Docs API:** `localhost:8000/docs`")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**📡 Estado del Sistema**")
        if health:
            badge_html("✅ API Operativa", "success")
        else:
            badge_html("❌ API Desconectada", "error")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Módulos del Proyecto")
    mods = [
        ("1","📈","Análisis Técnico","SMA, EMA, RSI, MACD, Bollinger"),
        ("2","📉","Rendimientos","Estadísticas, Normalidad, Q-Q"),
        ("3","🌊","ARCH/GARCH","Volatilidad Condicional"),
        ("4","⚖️","CAPM y Beta","Regresión, Beta, Retorno"),
        ("5","🛡️","VaR y CVaR","Param., Histórico, MC"),
        ("6","🎯","Markowitz","Frontera Eficiente"),
        ("7","🚦","Señales ★","Panel Semáforo"),
        ("8","🌍","Macro ★","Alpha, Tracking, Benchmark"),
    ]
    cols = st.columns(4)
    for i, (num, icon, name, desc) in enumerate(mods):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:18px 12px;">
                <div style="font-size:26px;">{icon}</div>
                <div style="font-weight:700;color:#6366F1;font-size:12px;margin-top:4px;">Módulo {num}</div>
                <div style="font-weight:600;color:#1E293B;margin:4px 0;font-size:14px;">{name}</div>
                <div style="font-size:11px;color:#94A3B8;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 1
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "📈 Módulo 1 · Técnico":
    page_header("📈 Módulo 1 · Análisis Técnico",
                "SMA, EMA, RSI, MACD y Bandas de Bollinger")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        ticker  = st.selectbox("Activo", list(TICKERS.keys()),
                               format_func=lambda t: f"{t} — {TICKERS[t]}")
        periodo = st.selectbox("Horizonte", ["1y","2y","5y"])
        sma_c   = st.slider("SMA Corto", 5, 50, 20)
        sma_l   = st.slider("SMA Largo", 21, 200, 50)
        rsi_p   = st.slider("Período RSI", 5, 30, 14)
        calcular = st.button("🔄 Calcular", type="primary", use_container_width=True)

    if calcular:
        with st.spinner("Consultando backend..."):
            data = api_post("/api/tecnico/indicadores", {
                "ticker": ticker, "periodo": periodo,
                "sma_corto": sma_c, "sma_largo": sma_l,
                "rsi_periodo": rsi_p, "bb_periodo": 20, "bb_std": 2.0
            })
        if data:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("💵 Precio Actual",   f"${data['ultimo_precio']:,.2f}")
            m2.metric("📈 Retorno Período", f"{data['retorno_periodo_pct']:.2f}%",
                      delta=f"{data['retorno_periodo_pct']:.2f}%")
            m3.metric("⚡ RSI Actual",       f"{data['rsi_actual']:.1f}")
            m4.metric("📊 Volatilidad",      f"{data['volatilidad_diaria_pct']:.2f}%")

            df = pd.DataFrame(data["datos"])
            df["fecha"] = pd.to_datetime(df["fecha"])

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.72, 0.28], vertical_spacing=0.04,
                                subplot_titles=("Precio e Indicadores", "RSI"))
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["close"],
                name="Precio", line=dict(color=PRIMARY, width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["fecha"], y=df[f"sma_{sma_c}"],
                name=f"SMA {sma_c}", line=dict(color=WARNING, width=1.5, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["fecha"], y=df[f"sma_{sma_l}"],
                name=f"SMA {sma_l}", line=dict(color=DANGER, width=1.5, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["bb_upper"],
                name="BB+", line=dict(color=MUTED, width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["bb_lower"],
                name="BB−", line=dict(color=MUTED, width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(148,163,184,0.08)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["fecha"], y=df["rsi"],
                name="RSI", line=dict(color="#8B5CF6", width=2)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color=DANGER,  row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=SUCCESS, row=2, col=1)
            fig.update_layout(height=600, hovermode="x unified", **PLOT_TPL)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📘 Interpretación"):
                cc = st.columns(3)
                rsi_v = data['rsi_actual']
                estado_rsi = "sobrecompra" if rsi_v > 70 else "sobreventa" if rsi_v < 30 else "zona neutral"
                with cc[0]:
                    st.info(f"**SMA:** Cruce SMA{sma_c} sobre SMA{sma_l} → tendencia alcista.")
                with cc[1]:
                    st.info(f"**RSI = {rsi_v:.1f}:** Activo en {estado_rsi}.")
                with cc[2]:
                    st.info("**Bollinger:** Expansión de bandas indica mayor volatilidad.")


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 2
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "📉 Módulo 2 · Rendimientos":
    page_header("📉 Módulo 2 · Rendimientos",
                "Caracterización estadística y pruebas de normalidad")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        ticker  = st.selectbox("Activo", list(TICKERS.keys()), key="m2_t")
        periodo = st.selectbox("Horizonte", ["1y","2y","5y"], index=1, key="m2_p")
        tipo    = st.radio("Tipo de rendimiento", ["Simple","Logarítmico"])
        calcular = st.button("🔄 Calcular", type="primary", use_container_width=True)

    if calcular:
        with st.spinner("Consultando backend..."):
            data = api_post("/api/rendimientos/estadisticas",
                            {"ticker": ticker, "periodo": periodo, "tipo": tipo})
        if data:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("μ Media",           f"{data['media']:.5f}")
            m2.metric("σ Desv. Estándar",  f"{data['desviacion']:.5f}")
            m3.metric("Asimetría (Skew)",  f"{data['asimetria']:.3f}")
            m4.metric("Curtosis (Exceso)", f"{data['curtosis']:.3f}")

            rend   = [r["rendimiento"] for r in data["datos_rendimientos"]]
            fechas = [r["fecha"]       for r in data["datos_rendimientos"]]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Histograma + Curva Normal Teórica",
                                "Gráfico Q-Q vs Normal",
                                "Boxplot de Rendimientos",
                                "Serie Temporal (Volatility Clustering)"),
                vertical_spacing=0.14, horizontal_spacing=0.1
            )
            # Histograma
            fig.add_trace(go.Histogram(x=rend, histnorm="probability density",
                name="Rendimientos", marker_color=PRIMARY, opacity=0.65), row=1, col=1)
            x_r = np.linspace(min(rend), max(rend), 200)
            y_n = norm.pdf(x_r, data["media"], data["desviacion"])
            fig.add_trace(go.Scatter(x=x_r, y=y_n, mode="lines",
                name="Normal Teórica", line=dict(color=DANGER, width=2.5)), row=1, col=1)
            # Q-Q
            qq = sm2.ProbPlot(np.array(rend), dist=norm, fit=True)
            tq, sq = qq.theoretical_quantiles, np.sort(rend)
            fig.add_trace(go.Scatter(x=tq, y=sq, mode="markers",
                name="Q-Q", marker=dict(size=3, color=PRIMARY, opacity=0.5)), row=1, col=2)
            lr = [float(min(tq)), float(max(tq))]
            fig.add_trace(go.Scatter(x=lr, y=lr, mode="lines",
                name="45°", line=dict(color=DANGER, dash="dash", width=2)), row=1, col=2)
            # Boxplot
            fig.add_trace(go.Box(y=rend, name="Rend.",
                boxpoints="outliers", marker_color=PRIMARY, line_color=PRIMARY), row=2, col=1)
            # Serie temporal
            fig.add_trace(go.Scatter(x=fechas, y=rend, mode="lines",
                name="Retornos", line=dict(width=1, color=PRIMARY)), row=2, col=2)
            fig.update_layout(height=680, showlegend=False, **PLOT_TPL)
            st.plotly_chart(fig, use_container_width=True)

            # ── Pruebas de Normalidad (BUG CORREGIDO) ──
            st.markdown("### 🔬 Pruebas de Normalidad")
            c1, c2 = st.columns(2)

            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**Jarque-Bera** · p-valor: `{data['jarque_bera_pvalor']:.6f}`")
                st.markdown("H₀: La distribución es Normal")
                if data["es_normal_jb"]:
                    badge_html("✅ No rechazamos H₀ · Distribución Normal", "success")
                else:
                    badge_html("❌ Rechazamos H₀ · No sigue distribución Normal", "error")
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**Shapiro-Wilk** · p-valor: `{data['shapiro_pvalor']:.6f}`")
                st.markdown("H₀: La distribución es Normal")
                if data["es_normal_sw"]:
                    badge_html("✅ No rechazamos H₀ · Distribución Normal", "success")
                else:
                    badge_html("❌ Rechazamos H₀ · No sigue distribución Normal", "error")
                st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("📖 Discusión de Hechos Estilizados"):
                kurt = data['curtosis']
                asim = data['asimetria']
                st.markdown(f"""
**Colas Pesadas (Leptocurtosis):** Curtosis de exceso = **{kurt:.2f}**.
{"Las colas son significativamente más anchas que la normal — mayor probabilidad de eventos extremos." if abs(kurt) > 1 else "Distribución cercana a la normal en curtosis."}

**Agrupamiento de Volatilidad:** La serie temporal muestra bloques de alta y baja volatilidad, motivando el uso de modelos GARCH.

**Efecto Apalancamiento:** Asimetría = **{asim:.2f}** → {"sesgo negativo: retornos negativos más extremos." if asim < -0.2 else "sesgo positivo." if asim > 0.2 else "distribución aproximadamente simétrica."}
                """)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 3
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🌊 Módulo 3 · ARCH/GARCH":
    page_header("🌊 Módulo 3 · ARCH/GARCH",
                "Modelado de volatilidad condicional y pronóstico de riesgo")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        ticker    = st.selectbox("Activo", list(TICKERS.keys()), key="m3_t")
        dist      = st.selectbox("Distribución", ["Normal","t-Student","Skewed t-Student"], index=1)
        horizonte = st.slider("Días a pronosticar", 5, 30, 10)
        calcular  = st.button("🔄 Ajustar Modelos", type="primary", use_container_width=True)

    with st.expander("💡 Justificación: ¿Por qué volatilidad condicional?", expanded=False):
        st.markdown("""
Los modelos **ARCH/GARCH** capturan el *agrupamiento de volatilidad* en series financieras.
- **ARCH(1):** La varianza depende del error cuadrado del período anterior.
- **GARCH(1,1):** Añade la varianza condicional rezagada — más parsimónico.
- **EGARCH(1,1):** Captura asimetría (efecto apalancamiento) sin restricción de positividad.
        """)

    if calcular:
        with st.spinner("Ajustando modelos GARCH (~30s)..."):
            data = api_post("/api/garch/volatilidad",
                            {"ticker": ticker, "horizonte": horizonte, "distribucion": dist})
        if data:
            st.markdown("### 📊 Comparativa de Modelos")
            df_mod = pd.DataFrame(data["comparativa_modelos"])
            cols_num = [c for c in ["log_likelihood","aic","bic"] if c in df_mod.columns]
            if cols_num:
                styled = df_mod.set_index("modelo").style
                if "aic" in cols_num and "bic" in cols_num:
                    styled = styled.highlight_min(subset=["aic","bic"], color="#BBF7D0")
                if "log_likelihood" in cols_num:
                    styled = styled.highlight_max(subset=["log_likelihood"], color="#BBF7D0")
                st.dataframe(styled.format({c: "{:.2f}" for c in cols_num}),
                             use_container_width=True)
                st.caption("✅ Verde = mejor valor (AIC/BIC menores; Log-Likelihood mayor)")

            st.markdown("### 🔍 Diagnóstico GARCH(1,1)")
            col1, col2 = st.columns([3, 1])
            with col1:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(y=data["residuos_std"], mode="lines",
                    name="Residuos Estandarizados", line=dict(color=PRIMARY, width=1.2)))
                fig_r.add_hline(y= 2, line_dash="dash", line_color=DANGER, opacity=0.5)
                fig_r.add_hline(y=-2, line_dash="dash", line_color=DANGER, opacity=0.5)
                fig_r.update_layout(title="Residuos Estandarizados · GARCH(1,1)",
                                    yaxis_title="Residuo Std", height=320, **PLOT_TPL)
                st.plotly_chart(fig_r, use_container_width=True)
            with col2:
                pval = data["jb_residuos_pvalor"]
                st.metric("JB p-valor (residuos)", f"{pval:.4f}")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                if pval < 0.05:
                    badge_html("⚠️ Residuos no normales", "warning")
                    st.markdown("<br><small>Frecuente en finanzas — usar distribución t-Student.</small>",
                                unsafe_allow_html=True)
                else:
                    badge_html("✅ Residuos normales", "success")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"### 🔮 Pronóstico de Volatilidad · {horizonte} días")
            vol_fc = data["pronostico_volatilidad"]
            dias   = list(range(1, horizonte + 1))
            fig_f  = go.Figure()
            fig_f.add_trace(go.Scatter(x=dias, y=vol_fc, mode="lines+markers",
                name="Vol. Pronosticada (%)",
                line=dict(color=PRIMARY, width=2.5),
                marker=dict(size=7, color=PRIMARY),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"))
            fig_f.update_layout(xaxis_title="Días", yaxis_title="Volatilidad (%)",
                                height=340, **PLOT_TPL)
            st.plotly_chart(fig_f, use_container_width=True)
            st.info(f"Vol. promedio pronosticada: **{np.mean(vol_fc):.3f}%** diario "
                    f"(**{np.mean(vol_fc)*np.sqrt(252):.2f}%** anualizado).")


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 4
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "⚖️ Módulo 4 · CAPM y Beta":
    page_header("⚖️ Módulo 4 · CAPM y Beta",
                "Línea característica del activo, Beta y retorno esperado CAPM")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        ticker  = st.selectbox("Activo", list(TICKERS.keys()), key="m4_t")
        periodo = st.selectbox("Horizonte", ["1y","2y"], index=1, key="m4_p")
        calcular = st.button("🔄 Calcular", type="primary", use_container_width=True)

    if calcular:
        with st.spinner("Consultando backend..."):
            data = api_post("/api/capm/beta",
                            {"ticker": ticker, "benchmark": "^GSPC", "periodo": periodo})
        if data:
            col1, col2 = st.columns([3, 1])
            with col1:
                df_r   = pd.DataFrame(data["datos_regresion"])
                x_line = np.array([df_r["market"].min(), df_r["market"].max()])
                alpha_r = data["retorno_esperado_pct"] - data["beta"] * data["rm_anual_pct"]
                y_line  = alpha_r + data["beta"] * x_line
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_r["market"], y=df_r["asset"], mode="markers",
                    name="Observaciones", marker=dict(color=PRIMARY, size=4, opacity=0.4)))
                fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                    name=f"Regresión (β={data['beta']:.2f})",
                    line=dict(color=DANGER, width=2.5)))
                fig.update_layout(title=f"Línea Característica {ticker} vs S&P 500",
                    xaxis_title="Rendimiento Mercado", yaxis_title=f"Rendimiento {ticker}",
                    height=430, **PLOT_TPL)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric("Beta (β)",        f"{data['beta']:.4f}")
                st.metric("Retorno CAPM",    f"{data['retorno_esperado_pct']:.2%}")
                st.metric("R²",              f"{data['r_squared']:.4f}")
                st.metric("Rf (10Y)",        f"{data['rf_anual_pct']:.2%}")
                st.metric("Rm Anual",        f"{data['rm_anual_pct']:.2%}")
                st.markdown("---")
                cls = data["clasificacion"]
                if cls == "Agresivo":
                    badge_html("🔴 Agresivo (β > 1.1)", "error")
                elif cls == "Defensivo":
                    badge_html("🟢 Defensivo (β < 0.9)", "success")
                else:
                    badge_html("🔵 Neutro (β ≈ 1)", "info")
                st.markdown('</div>', unsafe_allow_html=True)

            beta = data["beta"]
            st.info(
                f"📌 Con β = {beta:.2f}, el activo **{'amplifica' if beta > 1 else 'amortigua'}** "
                f"los movimientos del mercado un **{abs((beta-1)*100):.1f}%**. "
                f"Retorno esperado CAPM: **{data['retorno_esperado_pct']:.2%}** anual."
            )


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 5
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🛡️ Módulo 5 · VaR y CVaR":
    page_header("🛡️ Módulo 5 · VaR y CVaR",
                "Pérdida potencial máxima: Paramétrico, Histórico y Montecarlo")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        ticker    = st.selectbox("Activo", list(TICKERS.keys()), key="m5_t")
        confianza = st.select_slider("Nivel de Confianza", [0.95, 0.99], value=0.95)
        inversion = st.number_input("Inversión (USD)", value=10000, step=1000)
        calcular  = st.button("🔄 Calcular", type="primary", use_container_width=True)

    if calcular:
        with st.spinner("Calculando métricas de riesgo..."):
            data = api_post("/api/var/calcular", {
                "ticker": ticker, "confianza": confianza,
                "inversion": inversion, "n_sims": 10000
            })
        if data:
            rend  = data["datos_rendimientos"]
            vhist = data["var_historico_diario_pct"]
            cvar  = data["cvar_diario_pct"]

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=rend, histnorm="probability density",
                name="Distribución", marker_color=PRIMARY, opacity=0.6, nbinsx=80))
            x_loss = [v for v in rend if v <= vhist]
            fig.add_trace(go.Histogram(x=x_loss, histnorm="probability density",
                name="Cola de pérdida", marker_color=DANGER, opacity=0.5, nbinsx=30))
            fig.add_vline(x=vhist, line_dash="dash", line_color=DANGER, line_width=2,
                          annotation_text=f"VaR {confianza:.0%}", annotation_font_color=DANGER)
            fig.add_vline(x=cvar, line_dash="dot", line_color=WARNING, line_width=2,
                          annotation_text="CVaR", annotation_font_color=WARNING)
            fig.update_layout(title="Distribución de Rendimientos y Zonas de Riesgo",
                xaxis_title="Rendimiento Diario", yaxis_title="Densidad",
                height=380, **PLOT_TPL)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📋 Tabla Comparativa")
            tabla = pd.DataFrame({
                "Metodología": ["Paramétrico (Diario)","Paramétrico (Anual)",
                                "Simulación Histórica","Montecarlo (10k)","CVaR / Expected Shortfall"],
                "VaR / CVaR (%)": [
                    f"{data['var_parametrico_diario_pct']:.2%}",
                    f"{data['var_parametrico_anual_pct']:.2%}",
                    f"{data['var_historico_diario_pct']:.2%}",
                    f"{data['var_montecarlo_diario_pct']:.2%}",
                    f"{data['cvar_diario_pct']:.2%}",
                ],
                "Pérdida USD": [
                    f"${data['perdida_param_usd']:,.2f}", "—",
                    f"${data['perdida_hist_usd']:,.2f}",
                    f"${data['perdida_mc_usd']:,.2f}",
                    f"${data['perdida_cvar_usd']:,.2f}",
                ]
            })
            st.dataframe(tabla.set_index("Metodología"), use_container_width=True)
            st.info(
                f"📌 Con **{confianza:.0%}** de confianza, la pérdida máxima diaria para "
                f"**${inversion:,.0f}** es **${data['perdida_hist_usd']:,.2f}** (histórico). "
                f"CVaR (peor escenario promedio): **${data['perdida_cvar_usd']:,.2f}**."
            )


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 6
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🎯 Módulo 6 · Markowitz":
    page_header("🎯 Módulo 6 · Frontera Eficiente de Markowitz",
                "Optimización Media-Varianza: Máximo Sharpe y Mínima Varianza")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        tickers_sel = st.multiselect("Activos", list(TICKERS.keys()),
                                     default=list(TICKERS.keys())[:5])
        num_port    = st.select_slider("Simulaciones", [1000, 5000, 10000], value=5000)
        calcular    = st.button("🔄 Calcular Frontera", type="primary", use_container_width=True)

    if calcular:
        if len(tickers_sel) < 2:
            st.warning("Selecciona al menos 2 activos.")
        else:
            with st.spinner("Simulando portafolios Montecarlo..."):
                data = api_post("/api/markowitz/frontera", {
                    "tickers": tickers_sel, "num_portafolios": num_port, "periodo": "2y"
                })
            if data:
                st.markdown("### 🔗 Matriz de Correlación")
                corr_df = pd.DataFrame(data["matriz_correlacion"])
                fig_c = px.imshow(corr_df, text_auto=".2f",
                                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                fig_c.update_layout(height=340, **PLOT_TPL)
                st.plotly_chart(fig_c, use_container_width=True)

                st.markdown("### 📈 Frontera Eficiente")
                fe = pd.DataFrame(data["frontera_eficiente"])
                ms = data["portafolio_max_sharpe"]
                mv = data["portafolio_min_varianza"]

                fig_fe = px.scatter(fe, x="volatilidad", y="retorno", color="sharpe",
                    color_continuous_scale="Viridis", opacity=0.6,
                    labels={"volatilidad":"Riesgo (σ anual)","retorno":"Retorno anual","sharpe":"Sharpe"})
                fig_fe.add_trace(go.Scatter(
                    x=[ms["volatilidad_anual_pct"]], y=[ms["retorno_anual_pct"]],
                    mode="markers", name="Máx. Sharpe",
                    marker=dict(color=WARNING, size=16, symbol="star",
                                line=dict(color="#1E293B", width=1))))
                fig_fe.add_trace(go.Scatter(
                    x=[mv["volatilidad_anual_pct"]], y=[mv["retorno_anual_pct"]],
                    mode="markers", name="Mín. Varianza",
                    marker=dict(color=SUCCESS, size=14, symbol="diamond",
                                line=dict(color="#1E293B", width=1))))
                fig_fe.update_layout(height=440, **PLOT_TPL)
                st.plotly_chart(fig_fe, use_container_width=True)

                c1, c2 = st.columns(2)
                for col, port, label, color in [
                    (c1, ms, "⭐ Máximo Sharpe",    "#6366F133"),
                    (c2, mv, "💎 Mínima Varianza",  "#22C55E33")
                ]:
                    with col:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(f"**{label}**")
                        st.metric("Sharpe",     f"{port['sharpe_ratio']:.4f}")
                        st.metric("Retorno",    f"{port['retorno_anual_pct']:.2%}")
                        st.metric("Volatilidad",f"{port['volatilidad_anual_pct']:.2%}")
                        pesos_df = pd.DataFrame(list(port["pesos"].items()),
                                                columns=["Activo","Peso"]).set_index("Activo")
                        st.dataframe(pesos_df.style.format("{:.2%}").bar(color=color),
                                     use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 7
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🚦 Módulo 7 · Señales ★":
    page_header("🚦 Módulo 7 · Panel de Señales Algorítmicas",
                "Sistema automático de alertas basado en indicadores técnicos")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        ticker   = st.selectbox("Activo", list(TICKERS.keys()), key="m7_t")
        rsi_up   = st.slider("Sobrecompra RSI", 60, 80, 70)
        rsi_down = st.slider("Sobreventa RSI",  20, 40, 30)
        calcular = st.button("🔄 Actualizar", type="primary", use_container_width=True)

    if calcular:
        with st.spinner("Generando señales..."):
            data = api_post("/api/senales/panel",
                            {"ticker": ticker, "rsi_up": rsi_up, "rsi_down": rsi_down})
        if data:
            m1, m2, m3 = st.columns(3)
            m1.metric("💵 Precio Actual", f"${data['precio_actual']:,.2f}")
            m2.metric("⚡ RSI Actual",    f"{data['rsi_actual']:.2f}")
            m3.metric("📡 Señales",       str(len(data["senales"])))

            # Señal global (BUG CORREGIDO — sin inline st.success/error)
            gs = data["señal_global"]
            if gs == "COMPRA":
                st.markdown(f'<div class="signal-buy">🟢 SEÑAL GLOBAL: {gs}</div>',
                            unsafe_allow_html=True)
            elif gs == "VENTA":
                st.markdown(f'<div class="signal-sell">🔴 SEÑAL GLOBAL: {gs}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="signal-neut">🔵 SEÑAL GLOBAL: {gs}</div>',
                            unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Tarjetas individuales (BUG CORREGIDO — HTML puro, sin st.success/error inline)
            emoji_map = {"green":"🟢","red":"🔴","blue":"🔵"}
            cols = st.columns(len(data["senales"]))
            for i, sig in enumerate(data["senales"]):
                with cols[i]:
                    st.markdown(f"""
                    <div class="sig-card">
                        <div class="sig-label">{sig["indicador"].upper()}</div>
                        <div style="font-size:24px;">{emoji_map.get(sig["color"],"🔵")}</div>
                        <div class="sig-estado">{sig["estado"]}</div>
                        <div class="sig-desc">{sig["descripcion"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            badge_html("⚠️ Señales automáticas — no constituyen recomendación de inversión.", "warning")


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 8
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🌍 Módulo 8 · Macro ★":
    page_header("🌍 Módulo 8 · Macro y Benchmark",
                "Contexto macroeconómico y comparativa portafolio vs S&P 500")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros")
        tickers_sel = st.multiselect("Activos del portafolio", list(TICKERS.keys()),
                                     default=list(TICKERS.keys())[:5])
        calcular    = st.button("🔄 Calcular", type="primary", use_container_width=True)

    st.markdown("### 🌐 Indicadores Macroeconómicos")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tasa Libre de Riesgo (10Y)", "4.32%",   "+0.05 bps")
    m2.metric("Inflación (CPI)",            "3.10%",   "-0.10%",  delta_color="inverse")
    m3.metric("TRM COP/USD",                "$4,000",  "+15")
    m4.metric("Fed Funds Rate",             "5.25%",   "Sin cambio")

    if calcular:
        if not tickers_sel:
            st.warning("Selecciona al menos un activo.")
        else:
            with st.spinner("Calculando comparativa..."):
                data = api_post("/api/macro/benchmark", {
                    "tickers": tickers_sel, "benchmark": "^GSPC", "periodo": "1y"
                })
            if data:
                port  = pd.DataFrame(data["portafolio_acumulado"])
                bench = pd.DataFrame(data["benchmark_acumulado"])

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=port["fecha"], y=port["valor"],
                    name="Mi Portafolio",
                    line=dict(color=PRIMARY, width=3),
                    fill="tozeroy", fillcolor="rgba(99,102,241,0.06)"))
                fig.add_trace(go.Scatter(x=bench["fecha"], y=bench["valor"],
                    name="Benchmark (S&P 500)",
                    line=dict(color=MUTED, width=2, dash="dash")))
                fig.add_hline(y=100, line_dash="dot", line_color=MUTED, opacity=0.4)
                fig.update_layout(
                    title="Rendimiento Acumulado Base 100 · Portafolio vs S&P 500",
                    xaxis_title="Fecha", yaxis_title="Valor (Base 100)",
                    height=420, **PLOT_TPL)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Alpha",           f"{data['alpha_pct']:.2%}",  delta=f"{data['alpha_pct']:.2%}")
                c2.metric("Tracking Error",  f"{data['tracking_error_pct']:.2%}")
                c3.metric("Info. Ratio",     f"{data['information_ratio']:.2f}")
                c4.metric("Máx. Drawdown",   f"{data['max_drawdown_pct']:.2%}", delta_color="inverse")

                cb1, cb2 = st.columns(2)
                with cb1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("**Métricas de Selección**")
                    st.table(pd.DataFrame({
                        "Métrica": ["Alpha de Jensen","Tracking Error","Information Ratio"],
                        "Valor":   [f"{data['alpha_pct']:.2%}",
                                    f"{data['tracking_error_pct']:.2%}",
                                    f"{data['information_ratio']:.2f}"]
                    }).set_index("Métrica"))
                    st.markdown('</div>', unsafe_allow_html=True)
                with cb2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("**Resumen de Desempeño**")
                    st.table(pd.DataFrame({
                        "Métrica": ["Retorno Anualizado","Volatilidad Anual","Máx. Drawdown"],
                        "Valor":   [f"{data['rendimiento_portafolio_pct']:.2%}",
                                    f"{data['volatilidad_anual_pct']:.2%}",
                                    f"{data['max_drawdown_pct']:.2%}"]
                    }).set_index("Métrica"))
                    st.markdown('</div>', unsafe_allow_html=True)

                # Interpretación (BUG CORREGIDO — sin inline st.success/warning)
                if data["alpha_pct"] > 0:
                    badge_html(
                        f"🚀 El portafolio superó al benchmark con Alpha = {data['alpha_pct']:.2%}. "
                        f"La selección activa generó valor respecto al S&P 500.",
                        "success"
                    )
                else:
                    badge_html(
                        f"📉 El portafolio sub-performó al benchmark (Alpha = {data['alpha_pct']:.2%}). "
                        f"Considera ajustar los pesos en Módulo 6 (Markowitz).",
                        "warning"
                    )