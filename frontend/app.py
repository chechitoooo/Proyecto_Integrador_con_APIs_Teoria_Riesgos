"""
frontend/app.py — Dashboard Financiero · Teoría de Riesgo
Ejecutar con: streamlit run frontend/app.py
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

st.set_page_config(
    page_title="Dashboard Financiero · Teoría de Riesgo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0F0C29 0%, #1a1a3e 50%, #0d0d2b 100%); background-attachment: fixed; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a0533 0%, #0d1b4f 100%); border-right: 1px solid rgba(139,92,246,0.3); }
[data-testid="stSidebar"] * { color: #E2D9F3 !important; }
[data-testid="stSidebar"] .stButton>button { background: linear-gradient(135deg, #7C3AED, #4F46E5) !important; color: white !important; border: none !important; border-radius: 12px !important; font-weight: 700 !important; padding: 12px !important; box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important; transition: all 0.3s ease !important; }
[data-testid="stSidebar"] .stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(124,58,237,0.6) !important; }
[data-testid="metric-container"] { background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03)); border: 1px solid rgba(139,92,246,0.35); border-radius: 16px; padding: 18px 22px; backdrop-filter: blur(10px); box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
[data-testid="stMetricLabel"] { color: #A78BFA !important; font-size: 12px !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #F8FAFC !important; font-size: 28px !important; font-weight: 800 !important; }
h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; font-family: 'Outfit', sans-serif !important; font-weight: 800 !important; }
p, li, span, label { color: #CBD5E1 !important; font-family: 'Outfit', sans-serif !important; }
.page-header { background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #0EA5E9 100%); padding: 28px 36px; border-radius: 20px; margin-bottom: 28px; box-shadow: 0 10px 40px rgba(124,58,237,0.4); }
.page-header h2 { color: white !important; margin: 0; font-size: 26px !important; font-weight: 800 !important; }
.page-header p { color: rgba(255,255,255,0.85) !important; margin: 6px 0 0; font-size: 14px !important; }
.card { background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); border: 1px solid rgba(139,92,246,0.25); border-radius: 20px; padding: 24px; margin-bottom: 16px; backdrop-filter: blur(12px); box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
.badge-success { background: rgba(34,197,94,0.15); color: #4ADE80; border: 1px solid rgba(74,222,128,0.4); border-radius: 12px; padding: 12px 18px; font-weight: 700; border-left: 4px solid #22C55E; }
.badge-error { background: rgba(239,68,68,0.15); color: #F87171; border: 1px solid rgba(248,113,113,0.4); border-radius: 12px; padding: 12px 18px; font-weight: 700; border-left: 4px solid #EF4444; }
.badge-info { background: rgba(59,130,246,0.15); color: #60A5FA; border: 1px solid rgba(96,165,250,0.4); border-radius: 12px; padding: 12px 18px; font-weight: 700; border-left: 4px solid #3B82F6; }
.badge-warning { background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(252,211,77,0.4); border-radius: 12px; padding: 12px 18px; font-weight: 700; border-left: 4px solid #F59E0B; }
.signal-buy { background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(16,185,129,0.1)); color: #4ADE80; border: 2px solid rgba(74,222,128,0.5); border-radius: 16px; padding: 22px; text-align: center; font-size: 24px; font-weight: 800; }
.signal-sell { background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1)); color: #F87171; border: 2px solid rgba(248,113,113,0.5); border-radius: 16px; padding: 22px; text-align: center; font-size: 24px; font-weight: 800; }
.signal-neut { background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(37,99,235,0.1)); color: #60A5FA; border: 2px solid rgba(96,165,250,0.5); border-radius: 16px; padding: 22px; text-align: center; font-size: 24px; font-weight: 800; }
.sig-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(139,92,246,0.3); border-radius: 16px; padding: 20px 14px; text-align: center; }
.sig-label { font-size: 10px; font-weight: 700; color: #A78BFA !important; letter-spacing: 1.5px; margin-bottom: 8px; }
.sig-estado { font-weight: 700; font-size: 16px; color: #F1F5F9 !important; margin-top: 8px; }
.sig-desc { font-size: 12px; color: #94A3B8 !important; margin-top: 5px; }
.status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; animation: pulse 2s infinite; }
.status-dot.online { background: #22C55E; box-shadow: 0 0 8px #22C55E; }
.status-dot.offline { background: #EF4444; box-shadow: 0 0 8px #EF4444; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.autoload-banner { background: linear-gradient(90deg, rgba(124,58,237,0.15), rgba(14,165,233,0.1)); border: 1px solid rgba(139,92,246,0.4); border-radius: 12px; padding: 14px 20px; margin-bottom: 20px; font-size: 13px; color: #C4B5FD !important; }
.ticker-chip { background: rgba(124,58,237,0.15); border: 1px solid rgba(139,92,246,0.4); border-radius: 8px; padding: 6px 12px; font-size: 12px; font-weight: 700; color: #C4B5FD !important; display: inline-block; margin: 2px; }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 4px; border: 1px solid rgba(139,92,246,0.2); }
.stTabs [data-baseweb="tab"] { border-radius: 10px; color: #A78BFA !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #7C3AED, #4F46E5) !important; color: white !important; }
hr { border-color: rgba(139,92,246,0.3) !important; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
PRIMARY = "#7C3AED"
ACCENT = "#4F46E5"
CYAN = "#0EA5E9"
SUCCESS = "#22C55E"
DANGER = "#EF4444"
WARNING = "#F59E0B"
INFO = "#3B82F6"
MUTED = "#64748B"
GOLD = "#F59E0B"

PLOT_TPL = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10,10,30,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="sans-serif", color="#CBD5E1", size=12),
    xaxis=dict(gridcolor="rgba(100,100,255,0.1)"),
    yaxis=dict(gridcolor="rgba(100,100,255,0.1)"),
)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def page_header(title, subtitle=""):
    st.markdown(
        f'<div class="page-header"><h2>{title}</h2>'
        + (f'<p>{subtitle}</p>' if subtitle else '') +
        '</div>', unsafe_allow_html=True
    )


def badge_html(text, kind="info"):
    st.markdown(f'<div class="badge-{kind}">{text}</div>', unsafe_allow_html=True)


def fmt_pval(p):
    if p is None:
        return "N/A"
    return f"{p:.2e}" if p < 0.0001 else f"{p:.4f}"


def section_title(icon, title):
    st.markdown(
        f'<h3 style="color:#C4B5FD;font-size:18px;font-weight:700;margin:24px 0 12px;">'
        f'{icon} {title}</h3>',
        unsafe_allow_html=True
    )


# ─── API ───────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"


def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
        if r.status_code == 200:
            return r.json()
        st.error(f"Error API ({r.status_code}): {r.json().get('detail', r.text)}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"No se puede conectar al backend: {API_BASE}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
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
    if data:
        return data.get("tickers", {})
    return {
        "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
        "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "NVIDIA",
        "JPM": "JPMorgan", "BAC": "Bank of America",
        "GLD": "Gold ETF", "BTC-USD": "Bitcoin",
    }


TICKERS = get_tickers()
TICKER_LIST = list(TICKERS.keys())


def build_ticker_selector(key_prefix, default_all=True):
    all_opt = "✅ Seleccionar todos"
    selected = st.multiselect(
        "Activos", [all_opt] + TICKER_LIST,
        default=[all_opt] if default_all else TICKER_LIST[:3],
        key=f"{key_prefix}_tickers"
    )
    return TICKER_LIST if (all_opt in selected or not selected) else selected


def process_tickers(tickers, api_endpoint, build_payload, key_prefix):
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, t in enumerate(tickers):
        status_text.text(f"Procesando {t} ({i+1}/{len(tickers)})...")
        data = api_post(api_endpoint, build_payload(t))
        if data:
            results[t] = data
            st.session_state[f"{key_prefix}_{t}"] = data
        progress_bar.progress((i + 1) / len(tickers))
    progress_bar.empty()
    status_text.empty()
    if results:
        st.success(f"Completado para {len(results)} activos")
    return results


def load_cached_results(tickers, key_prefix):
    results = {}
    for t in tickers:
        cached = st.session_state.get(f"{key_prefix}_{t}")
        if cached:
            results[t] = cached
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 10px 10px;">
        <div style="font-size:36px;">📊</div>
        <h1 style="font-size:20px;font-weight:900;color:white;margin:0;">Dashboard Financiero</h1>
        <p style="font-size:11px;color:#A78BFA;margin:4px 0 0;">Teoria de Riesgo · 2026</p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    opcion = st.radio(
        "Modulo",
        [
            "🏠 Home",
            "📈 Modulo 1 - Tecnico",
            "📉 Modulo 2 - Rendimientos",
            "🌊 Modulo 3 - GARCH",
            "⚖️ Modulo 4 - CAPM",
            "🛡️ Modulo 5 - VaR",
            "🎯 Modulo 6 - Markowitz",
            "🚦 Modulo 7 - Senales",
            "🌍 Modulo 8 - Macro",
            "📐 Modulo 9 - Renta Fija",
            "🎲 Modulo 10 - Opciones",
            "⚠️ Modulo 11 - Stress Testing",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    health = api_get("/api/utils/health")
    if health:
        st.markdown('<div class="badge-success"><span class="status-dot online"></span>Backend conectado</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-error"><span class="status-dot offline"></span>Backend desconectado</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════════════
if opcion == "🏠 Home":
    st.markdown("""
    <div style="background:linear-gradient(135deg,#7C3AED 0%,#4F46E5 40%,#0EA5E9 100%);
                border-radius:24px;padding:48px 44px;margin-bottom:32px;
                box-shadow:0 20px 60px rgba(124,58,237,0.45);position:relative;overflow:hidden;">
        <div style="position:absolute;top:-60px;right:-40px;width:260px;height:260px;
                    background:rgba(255,255,255,0.04);border-radius:50%;"></div>
        <h1 style="color:white;margin:0;font-size:38px;font-weight:900;">
            Dashboard de Analisis Financiero
        </h1>
        <p style="color:rgba(255,255,255,0.85);margin-top:12px;font-size:16px;">
            Proyecto Final · Teoria de Riesgo · Analisis cuantitativo de riesgos y activos financieros
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
            <h3 style="color:#C4B5FD;">Integrantes</h3>
            <p>Sergio David Huertas Ramirez</p>
            <p>Sergio Andres Prieto Orjuela</p>
            <hr style="border-color:rgba(139,92,246,0.3);">
            <p style="font-size:13px;color:#94A3B8;">Materia: <b style="color:#C4B5FD;">Teoria de Riesgo</b></p>
            <p style="font-size:13px;color:#94A3B8;">Profesor: <b style="color:#C4B5FD;">Javier Sierra</b></p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <h3 style="color:#C4B5FD;">Arquitectura</h3>
            <p><span style="background:rgba(124,58,237,0.25);border-radius:6px;padding:3px 8px;color:#C4B5FD;font-weight:700;">Frontend</span> Streamlit</p>
            <p><span style="background:rgba(14,165,233,0.2);border-radius:6px;padding:3px 8px;color:#7DD3FC;font-weight:700;">Backend</span> FastAPI REST</p>
            <p><span style="background:rgba(34,197,94,0.2);border-radius:6px;padding:3px 8px;color:#86EFAC;font-weight:700;">HTTP</span> requests</p>
            <p><span style="background:rgba(245,158,11,0.2);border-radius:6px;padding:3px 8px;color:#FCD34D;font-weight:700;">Docs API</span> /docs</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h3 style="color:#C4B5FD;">Estado del Sistema</h3>', unsafe_allow_html=True)
        if health:
            badge_html("API Operativa — Backend OK", "success")
        else:
            badge_html("API Desconectada", "error")
        st.markdown(f"""
        <p style="margin-top:14px;font-size:12px;color:#64748B;">
            Activos disponibles: <b style="color:#C4B5FD;">{len(TICKERS)}</b><br>
            Modulos activos: <b style="color:#C4B5FD;">11</b>
        </p>
        </div>
        """, unsafe_allow_html=True)

    section_title("Modulos del Proyecto", "")
    mods = [
        ("1", "📈", "Tecnico", "SMA · RSI · Bollinger"),
        ("2", "📉", "Rendimientos", "Estadisticas · Normalidad"),
        ("3", "🌊", "ARCH/GARCH", "Volatilidad condicional"),
        ("4", "⚖️", "CAPM y Beta", "Linea caracteristica"),
        ("5", "🛡️", "VaR y CVaR", "Riesgo de perdida"),
        ("6", "🎯", "Markowitz", "Frontera eficiente"),
        ("7", "🚦", "Senales", "Panel semaforo"),
        ("8", "🌍", "Macro", "Alpha · Benchmark"),
        ("9", "📐", "Renta Fija", "Curva · Duracion"),
        ("10", "🎲", "Opciones", "Black-Scholes · Greeks"),
        ("11", "⚠️", "Stress Testing", "Escenarios extremos"),
    ]
    cols = st.columns(4)
    for i, (num, icon, name, desc) in enumerate(mods):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(255,255,255,0.07),rgba(255,255,255,0.02));
                        border:1px solid rgba(139,92,246,0.3);border-radius:18px;padding:22px 16px;
                        text-align:center;margin-bottom:12px;">
                <div style="font-size:32px;">{icon}</div>
                <div style="font-size:10px;font-weight:700;color:#A78BFA;">Modulo {num}</div>
                <div style="font-size:15px;font-weight:700;color:#F1F5F9;">{name}</div>
                <div style="font-size:11px;color:#94A3B8;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    section_title("Activos Disponibles", "")
    chips = " ".join([f'<span class="ticker-chip">{t}</span>' for t in TICKER_LIST])
    st.markdown(f'<div style="margin-bottom:20px;">{chips}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 1 — TECNICO
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "📈 Modulo 1 - Tecnico":
    page_header("Modulo 1 · Analisis Tecnico", "SMA, EMA, RSI, MACD, Bollinger y Estocastico")

    with st.sidebar:
        st.markdown("### Parametros")
        tickers = build_ticker_selector("m1")
        periodo = st.selectbox("Horizonte", ["1y", "2y", "5y"], index=1, key="m1_p")
        tipo_graf = st.radio("Tipo de grafico", ["Linea", "Velas japonesas"], key="m1_tipo", horizontal=True)
        sma_c = st.slider("SMA Corto", 5, 50, 20, key="m1_smac")
        sma_l = st.slider("SMA Largo", 21, 200, 50, key="m1_smal")
        ema_p = st.slider("Periodo EMA", 5, 50, 21, key="m1_ema")
        rsi_p = st.slider("Periodo RSI", 5, 30, 14, key="m1_rsi")
        stoch_k = st.slider("Estocastico %K", 5, 21, 14, key="m1_stochk")
        calcular = st.button("Calcular", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">Analizando {len(tickers)} activos · Horizonte: {periodo}</div>', unsafe_allow_html=True)

    results = {}
    if calcular:
        results = process_tickers(tickers, "/api/tecnico/indicadores",
            lambda t: {"ticker": t, "periodo": periodo, "sma_corto": sma_c, "sma_largo": sma_l,
                       "ema_periodo": ema_p, "rsi_periodo": rsi_p, "bb_periodo": 20, "bb_std": 2.0,
                       "stoch_k": stoch_k}, "m1")
    if not results:
        results = load_cached_results(tickers, "m1")

    if results:
        tabs = st.tabs([f"{t} · {TICKERS.get(t, t)}" for t in results])
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### {TICKERS.get(ticker, ticker)} (`{ticker}`)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Precio Actual", f"${data['ultimo_precio']:,.2f}")
                m2.metric("Retorno Periodo", f"{data['retorno_periodo_pct']:.2f}%")
                m3.metric("RSI Actual", f"{data['rsi_actual']:.1f}")
                m4.metric("Volatilidad", f"{data['volatilidad_diaria_pct']:.2f}%")

                df = pd.DataFrame(data["datos"])
                df["fecha"] = pd.to_datetime(df["fecha"])

                fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    row_heights=[0.45, 0.18, 0.18, 0.19], vertical_spacing=0.03,
                    subplot_titles=("Precio e Indicadores", "RSI", "MACD", "Estocastico"))

                if tipo_graf == "Velas japonesas" and all(c in df.columns for c in ["open", "high", "low", "close"]):
                    fig.add_trace(go.Candlestick(x=df["fecha"], open=df["open"], high=df["high"],
                        low=df["low"], close=df["close"], name="OHLC",
                        increasing_line_color=SUCCESS, decreasing_line_color=DANGER), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df["close"],
                        name="Precio", line=dict(color=PRIMARY, width=2.5)), row=1, col=1)

                if f"sma_{sma_c}" in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df[f"sma_{sma_c}"],
                        name=f"SMA {sma_c}", line=dict(color=WARNING, width=1.5, dash="dot")), row=1, col=1)
                if f"sma_{sma_l}" in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df[f"sma_{sma_l}"],
                        name=f"SMA {sma_l}", line=dict(color=DANGER, width=1.5, dash="dot")), row=1, col=1)
                if "ema" in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df["ema"],
                        name=f"EMA {ema_p}", line=dict(color=SUCCESS, width=1.5, dash="dashdot")), row=1, col=1)
                if "bb_upper" in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df["bb_upper"],
                        name="BB+", line=dict(color="#8B5CF6", width=1, dash="dot")), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df["bb_lower"],
                        name="BB-", line=dict(color="#8B5CF6", width=1, dash="dot"),
                        fill="tonexty", fillcolor="rgba(139,92,246,0.06)"), row=1, col=1)
                if "rsi" in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df["rsi"],
                        name="RSI", line=dict(color=CYAN, width=2)), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color=DANGER, row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color=SUCCESS, row=2, col=1)
                if "macd" in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df["macd"],
                        name="MACD", line=dict(color=PRIMARY, width=1.8)), row=3, col=1)
                if "macd_signal" in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df["macd_signal"],
                        name="Senal", line=dict(color=WARNING, width=1.5, dash="dot")), row=3, col=1)
                if "macd_hist" in df.columns:
                    colors_macd = [SUCCESS if v >= 0 else DANGER for v in df["macd_hist"]]
                    fig.add_trace(go.Bar(x=df["fecha"], y=df["macd_hist"],
                        name="Histograma", marker_color=colors_macd, opacity=0.7), row=3, col=1)
                fig.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.5, row=3, col=1)
                if "stoch_k" in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df["stoch_k"],
                        name="%K", line=dict(color=CYAN, width=1.8)), row=4, col=1)
                if "stoch_d" in df.columns:
                    fig.add_trace(go.Scatter(x=df["fecha"], y=df["stoch_d"],
                        name="%D", line=dict(color=WARNING, width=1.5, dash="dot")), row=4, col=1)
                    fig.add_hline(y=80, line_dash="dash", line_color=DANGER, opacity=0.6, row=4, col=1)
                    fig.add_hline(y=20, line_dash="dash", line_color=SUCCESS, opacity=0.6, row=4, col=1)

                fig.update_layout(height=800, hovermode="x unified", xaxis_rangeslider_visible=False, **PLOT_TPL)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Interpretacion de Indicadores"):
                    rsi_v = data['rsi_actual']
                    estado = "sobrecompra" if rsi_v > 70 else "sobreventa" if rsi_v < 30 else "neutral"
                    st.markdown(f"**RSI = {rsi_v:.1f}:** Activo en zona de {estado}.")
                    st.markdown("**SMA/EMA:** Cruce de SMA corta sobre larga indica tendencia alcista.")
                    st.markdown("**Bollinger:** Expansion de bandas = mayor volatilidad.")
                    st.markdown("**MACD:** Histograma verde = momentum positivo.")
                    st.markdown("**Estocastico:** %K cruzando %D en zona <20 = compra; >80 = venta.")
    else:
        st.info("Selecciona activos en el panel izquierdo y pulsa Calcular.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 2 — RENDIMIENTOS
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "📉 Modulo 2 - Rendimientos":
    page_header("Modulo 2 · Rendimientos", "Caracterizacion estadistica, pruebas de normalidad y hechos estilizados")

    with st.sidebar:
        st.markdown("### Parametros")
        tickers = build_ticker_selector("m2")
        periodo = st.selectbox("Horizonte", ["1y", "2y", "5y"], index=1, key="m2_p")
        tipo = st.radio("Tipo rendimiento", ["Simple", "Logaritmico"], key="m2_tipo")
        calcular = st.button("Calcular", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">Analisis de rendimientos para {len(tickers)} activos · Tipo: {tipo}</div>', unsafe_allow_html=True)

    results = {}
    if calcular:
        results = process_tickers(tickers, "/api/rendimientos/estadisticas",
            lambda t: {"ticker": t, "periodo": periodo, "tipo": tipo}, "m2")
    if not results:
        results = load_cached_results(tickers, "m2")

    if results:
        tabs = st.tabs([f"{t} · {TICKERS.get(t, t)}" for t in results])
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### {TICKERS.get(ticker, ticker)} (`{ticker}`)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Media", f"{data['media']:.5f}")
                m2.metric("Desv. Estandar", f"{data['desviacion']:.5f}")
                m3.metric("Asimetria", f"{data['asimetria']:.3f}")
                m4.metric("Curtosis", f"{data['curtosis']:.3f}")

                rend = [r["rendimiento"] for r in data["datos_rendimientos"]]
                fechas = [r["fecha"] for r in data["datos_rendimientos"]]

                fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Histograma + Normal Teorica", "Q-Q vs Normal",
                                    "Boxplot", "Serie Temporal"),
                    vertical_spacing=0.14, horizontal_spacing=0.1)
                fig.add_trace(go.Histogram(x=rend, histnorm="probability density",
                    name="Rendimientos", marker_color=PRIMARY, opacity=0.7), row=1, col=1)
                x_r = np.linspace(min(rend), max(rend), 200)
                y_n = norm.pdf(x_r, data["media"], data["desviacion"])
                fig.add_trace(go.Scatter(x=x_r, y=y_n, mode="lines",
                    name="Normal Teorica", line=dict(color=DANGER, width=2.5)), row=1, col=1)
                qq = sm2.ProbPlot(np.array(rend), dist=norm, fit=True)
                tq, sq = qq.theoretical_quantiles, np.sort(rend)
                fig.add_trace(go.Scatter(x=tq, y=sq, mode="markers",
                    name="Q-Q", marker=dict(size=3, color=CYAN, opacity=0.6)), row=1, col=2)
                lr = [float(min(tq)), float(max(tq))]
                fig.add_trace(go.Scatter(x=lr, y=lr, mode="lines",
                    name="45°", line=dict(color=DANGER, dash="dash", width=2)), row=1, col=2)
                fig.add_trace(go.Box(y=rend, name="Rend.",
                    boxpoints="outliers", marker_color=PRIMARY, line_color=CYAN), row=2, col=1)
                fig.add_trace(go.Scatter(x=fechas, y=rend, mode="lines",
                    name="Retornos", line=dict(width=1, color=PRIMARY)), row=2, col=2)
                fig.update_layout(height=680, showlegend=False, **PLOT_TPL)
                st.plotly_chart(fig, use_container_width=True)

                section_title("", "Pruebas de Normalidad")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Jarque-Bera** · p-valor: `{fmt_pval(data['jarque_bera_pvalor'])}`")
                    if data["es_normal_jb"]:
                        badge_html("No rechazamos H0 · Distribucion Normal", "success")
                    else:
                        badge_html("Rechazamos H0 · No sigue distribucion Normal", "error")
                with c2:
                    st.markdown(f"**Shapiro-Wilk** · p-valor: `{fmt_pval(data['shapiro_pvalor'])}`")
                    if data["es_normal_sw"]:
                        badge_html("No rechazamos H0 · Distribucion Normal", "success")
                    else:
                        badge_html("Rechazamos H0 · No sigue distribucion Normal", "error")
    else:
        st.info("Selecciona activos y pulsa Calcular.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 3 — GARCH
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🌊 Modulo 3 - GARCH":
    page_header("Modulo 3 · ARCH/GARCH", "Modelado de volatilidad condicional y pronostico de riesgo")

    with st.sidebar:
        st.markdown("### Parametros")
        tickers = build_ticker_selector("m3")
        dist = st.selectbox("Distribucion", ["Normal", "t-Student", "Skewed t-Student"], index=1, key="m3_dist")
        lambda_ewma = st.slider("Lambda EWMA", 0.80, 0.99, 0.94, step=0.01, key="m3_lam")
        horizonte = st.slider("Dias a pronosticar", 5, 30, 10, key="m3_hor")
        calcular = st.button("Ajustar Modelos", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">Modelos ARCH/GARCH para {len(tickers)} activos · Dist: {dist} · Horizonte: {horizonte} dias</div>', unsafe_allow_html=True)

    results = {}
    if calcular:
        results = process_tickers(tickers, "/api/garch/volatilidad",
            lambda t: {"ticker": t, "horizonte": horizonte, "distribucion": dist, "lambda_ewma": lambda_ewma}, "m3")
    if not results:
        results = load_cached_results(tickers, "m3")

    if results:
        tabs = st.tabs([f"{t} · {TICKERS.get(t, t)}" for t in results])
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### {TICKERS.get(ticker, ticker)} (`{ticker}`)")

                section_title("", f"EWMA (lambda={lambda_ewma})")
                if data.get("ewma_volatilidad") and data.get("vol_muestral_rodante"):
                    fechas_vol = data.get("fechas_vol", list(range(len(data["ewma_volatilidad"]))))
                    fig_ewma = go.Figure()
                    fig_ewma.add_trace(go.Scatter(x=fechas_vol, y=data["ewma_volatilidad"],
                        name=f"EWMA (lambda={lambda_ewma})", line=dict(color=PRIMARY, width=2.5)))
                    fig_ewma.add_trace(go.Scatter(x=fechas_vol, y=data["vol_muestral_rodante"],
                        name="Vol. Muestral Rodante (22d)", line=dict(color=MUTED, width=1.5, dash="dot")))
                    fig_ewma.update_layout(title="Volatilidad EWMA vs Muestral Rodante",
                        yaxis_title="Volatilidad diaria (%)", height=320, **PLOT_TPL)
                    st.plotly_chart(fig_ewma, use_container_width=True)

                section_title("", "Comparativa de Modelos")
                df_mod = pd.DataFrame(data["comparativa_modelos"])
                cols_num = [c for c in ["log_likelihood", "aic", "bic"] if c in df_mod.columns]
                if cols_num:
                    styled = df_mod.set_index("modelo").style
                    if "aic" in cols_num and "bic" in cols_num:
                        styled = styled.highlight_min(subset=["aic", "bic"], color="#1a3a1a")
                    if "log_likelihood" in cols_num:
                        styled = styled.highlight_max(subset=["log_likelihood"], color="#1a3a1a")
                    st.dataframe(styled.format({c: "{:.2f}" for c in cols_num}), use_container_width=True)

                section_title("", "Residuos Estandarizados GARCH(1,1)")
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(y=data["residuos_std"], mode="lines",
                    name="Residuos Estandarizados", line=dict(color=CYAN, width=1.2)))
                fig_r.add_hline(y=2, line_dash="dash", line_color=DANGER, opacity=0.5)
                fig_r.add_hline(y=-2, line_dash="dash", line_color=DANGER, opacity=0.5)
                fig_r.update_layout(title="Residuos Estandarizados", height=320, **PLOT_TPL)
                st.plotly_chart(fig_r, use_container_width=True)
                pval = data.get("jb_residuos_pvalor", 1)
                if pval < 0.05:
                    badge_html(f"Residuos no normales (JB p-valor={fmt_pval(pval)})", "warning")
                else:
                    badge_html(f"Residuos normales (JB p-valor={fmt_pval(pval)})", "success")

                section_title("", f"Pronostico de Volatilidad · {horizonte} dias")
                vol_fc = data["pronostico_volatilidad"]
                dias = list(range(1, horizonte + 1))
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=dias, y=vol_fc, mode="lines+markers",
                    name="Vol. Pronosticada (%)", line=dict(color=PRIMARY, width=2.5),
                    marker=dict(size=8, color=CYAN), fill="tozeroy",
                    fillcolor="rgba(124,58,237,0.1)"))
                fig_f.update_layout(xaxis_title="Dias hacia adelante", yaxis_title="Volatilidad (%)",
                    height=360, **PLOT_TPL)
                st.plotly_chart(fig_f, use_container_width=True)
                badge_html(f"Vol. promedio pronosticada: {np.mean(vol_fc):.3f}% diario — {np.mean(vol_fc)*np.sqrt(252):.2f}% anualizado.", "info")

                section_title("", "Test ARCH-LM sobre Residuos")
                p_archlm = data.get("arch_lm_pvalor", None)
                if p_archlm is not None:
                    st.metric("ARCH-LM p-valor", fmt_pval(p_archlm))
                    if p_archlm < 0.05:
                        badge_html("Efectos ARCH residuales presentes — modelo podria mejorar", "warning")
                    else:
                        badge_html("No hay efectos ARCH residuales — modelo bien especificado", "success")
    else:
        st.info("Selecciona activos y pulsa Ajustar Modelos.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 4 — CAPM
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "⚖️ Modulo 4 - CAPM":
    page_header("Modulo 4 · CAPM y Beta", "Linea caracteristica del activo, Beta, Alpha de Jensen")

    with st.sidebar:
        st.markdown("### Parametros")
        tickers = build_ticker_selector("m4")
        periodo = st.selectbox("Horizonte", ["1y", "2y"], index=1, key="m4_p")
        calcular = st.button("Calcular", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">CAPM y Beta para {len(tickers)} activos vs S&P 500</div>', unsafe_allow_html=True)

    results = {}
    if calcular:
        results = process_tickers(tickers, "/api/capm/beta",
            lambda t: {"ticker": t, "benchmark": "^GSPC", "periodo": periodo}, "m4")
    if not results:
        results = load_cached_results(tickers, "m4")

    if results:
        tabs = st.tabs([f"{t} · {TICKERS.get(t, t)}" for t in results])
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### {TICKERS.get(ticker, ticker)} (`{ticker}`)")
                col1, col2 = st.columns([3, 1])
                with col1:
                    df_r = pd.DataFrame(data["datos_regresion"])
                    x_line = np.array([df_r["market"].min(), df_r["market"].max()])
                    alpha_r = data["retorno_esperado_pct"] - data["beta"] * data["rm_anual_pct"]
                    y_line = alpha_r + data["beta"] * x_line
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_r["market"], y=df_r["asset"], mode="markers",
                        name="Observaciones", marker=dict(color=PRIMARY, size=5, opacity=0.5)))
                    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                        name=f"Regresion (beta={data['beta']:.2f})", line=dict(color=CYAN, width=3)))
                    fig.update_layout(title=f"Linea Caracteristica {ticker} vs S&P 500",
                        xaxis_title="Rendimiento Mercado", yaxis_title=f"Rendimiento {ticker}",
                        height=430, **PLOT_TPL)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.metric("Beta", f"{data['beta']:.4f}")
                    st.metric("Retorno CAPM", f"{data['retorno_esperado_pct']:.2%}")
                    st.metric("R²", f"{data['r_squared']:.4f}")
                    st.metric("Rf (10Y)", f"{data['rf_anual_pct']:.2%}")
                    st.metric("Rm Anual", f"{data['rm_anual_pct']:.2%}")
                    cls = data.get("clasificacion", "Neutro")
                    cls_badge = {"Agresivo": "error", "Defensivo": "success", "Neutro": "info"}
                    badge_html(f"{cls} (beta={data['beta']:.2f})", cls_badge.get(cls, "info"))

                section_title("", "Alpha de Jensen")
                alpha_j = data.get("alpha_jensen_pct", None)
                if alpha_j is not None:
                    st.metric("Alpha de Jensen", f"{alpha_j:.4%}")
                    if alpha_j > 0:
                        badge_html(f"Alpha positivo ({alpha_j:.4%}): gestion activa anade valor", "success")
                    elif alpha_j < 0:
                        badge_html(f"Alpha negativo ({alpha_j:.4%}): rinde por debajo de CAPM", "error")
                    else:
                        badge_html("Alpha ≈ 0: consistente con CAPM", "info")

                st.info(f"Con beta = {data['beta']:.2f}, el activo {'amplifica' if data['beta'] > 1 else 'amortigua'} los movimientos del mercado un {abs((data['beta']-1)*100):.1f}%.")
    else:
        st.info("Selecciona activos y pulsa Calcular.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 5 — VaR
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🛡️ Modulo 5 - VaR":
    page_header("Modulo 5 · VaR y CVaR", "Perdida potencial maxima: Parametrico, Historico y Montecarlo")

    with st.sidebar:
        st.markdown("### Parametros")
        tickers = build_ticker_selector("m5")
        confianza = st.select_slider("Nivel de Confianza", [0.95, 0.99], value=0.95, key="m5_conf")
        inversion = st.number_input("Inversion (USD)", value=10000, step=1000, key="m5_inv")
        calcular = st.button("Calcular", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">VaR/CVaR para {len(tickers)} activos · Confianza: {confianza:.0%} · Inversion: ${inversion:,.0f}</div>', unsafe_allow_html=True)

    results = {}
    if calcular:
        results = process_tickers(tickers, "/api/var/calcular",
            lambda t: {"ticker": t, "confianza": confianza, "inversion": inversion, "n_sims": 10000}, "m5")
    if not results:
        results = load_cached_results(tickers, "m5")

    if results:
        tabs = st.tabs([f"{t} · {TICKERS.get(t, t)}" for t in results])
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### {TICKERS.get(ticker, ticker)} (`{ticker}`)")
                rend = data["datos_rendimientos"]
                vhist = data["var_historico_diario_pct"]
                cvar = data["cvar_diario_pct"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("VaR Param. Diario", f"{data['var_parametrico_diario_pct']:.2%}")
                m2.metric("VaR Historico", f"{vhist:.2%}")
                m3.metric("VaR Montecarlo", f"{data['var_montecarlo_diario_pct']:.2%}")
                m4.metric("CVaR", f"{cvar:.2%}")

                fig = go.Figure()
                fig.add_trace(go.Histogram(x=rend, histnorm="probability density",
                    name="Distribucion", marker_color=PRIMARY, opacity=0.65, nbinsx=80))
                x_loss = [v for v in rend if v <= vhist]
                fig.add_trace(go.Histogram(x=x_loss, histnorm="probability density",
                    name="Cola de perdida", marker_color=DANGER, opacity=0.6, nbinsx=30))
                fig.add_vline(x=vhist, line_dash="dash", line_color=DANGER, line_width=2.5,
                    annotation_text=f"VaR {confianza:.0%}")
                fig.add_vline(x=cvar, line_dash="dot", line_color=WARNING, line_width=2.5,
                    annotation_text="CVaR")
                fig.update_layout(title="Distribucion de Rendimientos y Zonas de Riesgo",
                    xaxis_title="Rendimiento Diario", yaxis_title="Densidad", height=400, **PLOT_TPL)
                st.plotly_chart(fig, use_container_width=True)

                section_title("", "Tabla Comparativa de Metodologias")
                tabla = pd.DataFrame({
                    "Metodologia": ["Parametrico (Diario)", "Parametrico (Anual)",
                        "Simulacion Historica", "Montecarlo (10k)", "CVaR / Expected Shortfall"],
                    "VaR / CVaR (%)": [
                        f"{data['var_parametrico_diario_pct']:.2%}",
                        f"{data['var_parametrico_anual_pct']:.2%}",
                        f"{vhist:.2%}",
                        f"{data['var_montecarlo_diario_pct']:.2%}",
                        f"{cvar:.2%}",
                    ],
                    "Perdida USD": [
                        f"${data['perdida_param_usd']:,.2f}", "-",
                        f"${data['perdida_hist_usd']:,.2f}",
                        f"${data['perdida_mc_usd']:,.2f}",
                        f"${data['perdida_cvar_usd']:,.2f}",
                    ]
                })
                st.dataframe(tabla.set_index("Metodologia"), use_container_width=True)

                section_title("", "Backtesting de Kupiec")
                if "excedencias_kupiec" in data:
                    T = len(data.get("datos_rendimientos", []))
                    kup_n = data["excedencias_kupiec"]
                    kup_lr = data["lr_uc_kupiec"]
                    kup_p = data["p_valor_kupiec"]
                    kup_ok = data["aprueba_kupiec"]
                    freq_obs = kup_n / T if T > 0 else 0
                    freq_teorica = 1 - confianza

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Excedencias", str(kup_n))
                    k2.metric("Dias (T)", str(T))
                    k3.metric("LR_POF", f"{kup_lr:.4f}")
                    k4.metric("p-valor Kupiec", fmt_pval(kup_p))

                    if not kup_ok:
                        badge_html(f"Kupiec rechaza el VaR (frec obs={freq_obs:.2%} vs teorica={freq_teorica:.2%})", "error")
                    else:
                        badge_html(f"Kupiec no rechaza el VaR al {confianza:.0%}", "success")
    else:
        st.info("Selecciona activos y pulsa Calcular.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 6 — MARKOWITZ
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🎯 Modulo 6 - Markowitz":
    page_header("Modulo 6 · Frontera Eficiente de Markowitz", "Optimizacion Media-Varianza")

    with st.sidebar:
        st.markdown("### Parametros")
        tickers_sel = st.multiselect("Activos", TICKER_LIST, default=TICKER_LIST[:5], key="m6_tickers")
        num_port = st.select_slider("Simulaciones", [1000, 5000, 10000], value=5000, key="m6_nport")
        no_short = st.checkbox("Sin ventas en corto (w >= 0)", value=True, key="m6_noshort")
        calcular = st.button("Calcular Frontera", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">Frontera eficiente para {len(tickers_sel)} activos</div>', unsafe_allow_html=True)

    if calcular:
        if len(tickers_sel) < 2:
            st.warning("Selecciona al menos 2 activos.")
        else:
            with st.spinner("Simulando portafolios..."):
                data = api_post("/api/markowitz/frontera", {
                    "tickers": tickers_sel, "num_portafolios": num_port,
                    "periodo": "2y", "no_short_selling": no_short
                })
            if data:
                st.session_state["m6_data"] = data

    data = st.session_state.get("m6_data")
    if data:
        tab1, tab2, tab3 = st.tabs(["Frontera Eficiente", "Correlaciones", "Portafolios Optimos"])

        ms = data["portafolio_max_sharpe"]
        mv = data["portafolio_min_varianza"]

        with tab1:
            fe = pd.DataFrame(data["frontera_eficiente"])
            fig_fe = px.scatter(fe, x="volatilidad", y="retorno", color="sharpe",
                color_continuous_scale="Plasma", opacity=0.5,
                labels={"volatilidad": "Riesgo (sigma anual)", "retorno": "Retorno anual", "sharpe": "Sharpe"})
            fig_fe.add_trace(go.Scatter(x=[ms["volatilidad_anual_pct"]], y=[ms["retorno_anual_pct"]],
                mode="markers", name="Max. Sharpe",
                marker=dict(color=GOLD, size=18, symbol="star", line=dict(color="white", width=1.5))))
            fig_fe.add_trace(go.Scatter(x=[mv["volatilidad_anual_pct"]], y=[mv["retorno_anual_pct"]],
                mode="markers", name="Min. Varianza",
                marker=dict(color=SUCCESS, size=15, symbol="diamond", line=dict(color="white", width=1.5))))
            fig_fe.update_layout(height=480, **PLOT_TPL)
            st.plotly_chart(fig_fe, use_container_width=True)

        with tab2:
            corr_df = pd.DataFrame(data["matriz_correlacion"])
            fig_c = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig_c.update_layout(height=400, **PLOT_TPL)
            st.plotly_chart(fig_c, use_container_width=True)

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Maximo Sharpe**")
                st.metric("Sharpe", f"{ms['sharpe_ratio']:.4f}")
                st.metric("Retorno", f"{ms['retorno_anual_pct']:.2%}")
                st.metric("Volatilidad", f"{ms['volatilidad_anual_pct']:.2%}")
                pesos_ms = pd.DataFrame(list(ms["pesos"].items()), columns=["Activo", "Peso"]).set_index("Activo")
                st.dataframe(pesos_ms.style.format("{:.2%}"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Minima Varianza**")
                st.metric("Sharpe", f"{mv['sharpe_ratio']:.4f}")
                st.metric("Retorno", f"{mv['retorno_anual_pct']:.2%}")
                st.metric("Volatilidad", f"{mv['volatilidad_anual_pct']:.2%}")
                pesos_mv = pd.DataFrame(list(mv["pesos"].items()), columns=["Activo", "Peso"]).set_index("Activo")
                st.dataframe(pesos_mv.style.format("{:.2%}"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Selecciona activos y pulsa Calcular Frontera.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 7 — SENALES
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🚦 Modulo 7 - Senales":
    page_header("Modulo 7 · Panel de Senales Algoritmicas", "Sistema automatico de alertas basado en indicadores tecnicos")

    with st.sidebar:
        st.markdown("### Parametros")
        tickers = build_ticker_selector("m7")
        rsi_up = st.slider("Sobrecompra RSI", 60, 80, 70, key="m7_rsiup")
        rsi_down = st.slider("Sobreventa RSI", 20, 40, 30, key="m7_rsido")
        calcular = st.button("Actualizar", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">Panel de senales para {len(tickers)} activos · RSI [{rsi_down}, {rsi_up}]</div>', unsafe_allow_html=True)

    results = {}
    if calcular:
        results = process_tickers(tickers, "/api/senales/panel",
            lambda t: {"ticker": t, "rsi_up": rsi_up, "rsi_down": rsi_down}, "m7")
    if not results:
        results = load_cached_results(tickers, "m7")

    if results:
        tabs = st.tabs([f"{t} · {TICKERS.get(t, t)}" for t in results])
        emoji_map = {"green": "🟢", "red": "🔴", "blue": "🔵"}
        for (ticker, data), tab in zip(results.items(), tabs):
            with tab:
                st.markdown(f"#### {TICKERS.get(ticker, ticker)} (`{ticker}`)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Precio Actual", f"${data['precio_actual']:,.2f}")
                m2.metric("RSI Actual", f"{data['rsi_actual']:.2f}")
                m3.metric("Senales", str(len(data["senales"])))

                gs = data["senal_global"]
                signal_class = {"COMPRA": "signal-buy", "VENTA": "signal-sell", "NEUTRAL": "signal-neut"}
                st.markdown(f'<div class="{signal_class.get(gs, "signal-neut")}">{gs}</div>', unsafe_allow_html=True)

                cols = st.columns(min(len(data["senales"]), 4))
                for i, sig in enumerate(data["senales"]):
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                        <div class="sig-card">
                            <div class="sig-label">{sig["indicador"].upper()}</div>
                            <div style="font-size:28px;">{emoji_map.get(sig["color"], "🔵")}</div>
                            <div class="sig-estado">{sig["estado"]}</div>
                            <div class="sig-desc">{sig["descripcion"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                badge_html("Senales automaticas — no constituyen recomendacion de inversion.", "warning")
    else:
        st.info("Selecciona activos y pulsa Actualizar.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 8 — MACRO
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🌍 Modulo 8 - Macro":
    page_header("Modulo 8 · Macro y Benchmark", "Contexto macroeconomico y comparativa portafolio vs S&P 500")

    with st.sidebar:
        st.markdown("### Parametros")
        tickers_sel = st.multiselect("Activos del portafolio", TICKER_LIST, default=TICKER_LIST[:5], key="m8_tickers")
        calcular = st.button("Calcular", type="primary", use_container_width=True)

    st.markdown('<div class="autoload-banner">Comparativa portafolio vs S&P 500</div>', unsafe_allow_html=True)

    macro_data = api_get("/api/macro/indicadores") or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tasa Libre Riesgo (10Y)", f"{macro_data.get('rf_10y_pct', 0):.2%}")
    m2.metric("Inflacion (CPI)", f"{macro_data.get('cpi_pct', 0):.2%}")
    m3.metric("TRM COP/USD", f"${macro_data.get('trm_cop_usd', 0):,.0f}")
    m4.metric("Fed Funds Rate", f"{macro_data.get('fed_funds_pct', 0):.2%}")

    if calcular:
        if not tickers_sel:
            st.warning("Selecciona al menos un activo.")
        else:
            with st.spinner("Calculando comparativa..."):
                data = api_post("/api/macro/benchmark", {
                    "tickers": tickers_sel, "benchmark": "^GSPC", "periodo": "1y"
                })
            if data:
                st.session_state["m8_data"] = data

    data = st.session_state.get("m8_data")
    if data:
        port = pd.DataFrame(data["portafolio_acumulado"])
        bench = pd.DataFrame(data["benchmark_acumulado"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port["fecha"], y=port["valor"],
            name="Mi Portafolio", line=dict(color=PRIMARY, width=3),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.08)"))
        fig.add_trace(go.Scatter(x=bench["fecha"], y=bench["valor"],
            name="Benchmark (S&P 500)", line=dict(color=MUTED, width=2, dash="dash")))
        fig.add_hline(y=100, line_dash="dot", line_color=MUTED, opacity=0.4)
        fig.update_layout(title="Rendimiento Acumulado Base 100 · Portafolio vs S&P 500",
            xaxis_title="Fecha", yaxis_title="Valor (Base 100)", height=440, **PLOT_TPL)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alpha", f"{data['alpha_pct']:.2%}")
        c2.metric("Tracking Error", f"{data['tracking_error_pct']:.2%}")
        c3.metric("Info. Ratio", f"{data['information_ratio']:.2f}")
        c4.metric("Max. Drawdown", f"{data['max_drawdown_pct']:.2%}")

        if data["alpha_pct"] > 0:
            badge_html(f"El portafolio supero al benchmark (Alpha = {data['alpha_pct']:.2%})", "success")
        else:
            badge_html(f"El portafolio sub-performo al benchmark (Alpha = {data['alpha_pct']:.2%})", "warning")
    else:
        st.info("Selecciona activos y pulsa Calcular.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 9 — RENTA FIJA
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "📐 Modulo 9 - Renta Fija":
    page_header("Modulo 9 · Renta Fija", "Curva de Rendimiento Nelson-Siegel y Duracion del Bono")

    with st.sidebar:
        st.markdown("### Activos")
        tickers_m9 = build_ticker_selector("m9")
        st.markdown("---")
        st.markdown("### Parametros Bono")
        cupon = st.number_input("Cupon anual (%)", value=5.0, step=0.25, key="m9_cupon")
        vencimiento = st.slider("Vencimiento (anos)", 1, 30, 10, key="m9_venc")
        valor_nominal = st.number_input("Valor Nominal", value=1000, step=100, key="m9_vn")
        calcular = st.button("Calcular Bono", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">Analisis de Renta Fija para <b>{len(tickers_m9)} activos</b>: {", ".join(tickers_m9[:6])}{"..." if len(tickers_m9) > 6 else ""}</div>', unsafe_allow_html=True)

    with st.spinner("Obteniendo curva de rendimientos..."):
        curva_data = api_get("/api/renta-fija/curva")

    if curva_data:
        fig_curva = go.Figure()
        if "puntos" in curva_data:
            df_puntos = pd.DataFrame(curva_data["puntos"])
            fig_curva.add_trace(go.Scatter(x=df_puntos["vencimiento"], y=df_puntos["tasa"],
                mode="markers", name="Tasas Observadas (FRED)", marker=dict(size=10, color=PRIMARY)))
        if "curva" in curva_data:
            df_modelo = pd.DataFrame(curva_data["curva"])
            fig_curva.add_trace(go.Scatter(x=df_modelo["vencimiento"], y=df_modelo["tasa"],
                mode="lines", name="Modelo Nelson-Siegel", line=dict(color=ACCENT, width=3, dash="dash")))
        fig_curva.update_layout(title="Curva de Rendimiento Spot",
            xaxis_title="Vencimiento (Anos)", yaxis_title="Tasa (%)", height=400, **PLOT_TPL)
        st.plotly_chart(fig_curva, use_container_width=True)

    if calcular:
        payload = {"cupon_pct": cupon, "vencimiento": vencimiento,
                   "valor_nominal": valor_nominal, "frecuencia": 2}
        bono_data = api_post("/api/renta-fija/bono", payload)
        if bono_data:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precio Bono", f"${bono_data.get('precio', 0):,.2f}")
            c2.metric("Duracion Macaulay", f"{bono_data.get('duracion_macaulay', 0):.2f} anos")
            c3.metric("Duracion Modificada", f"{bono_data.get('duracion_modificada', 0):.2f}")
            c4.metric("Convexidad", f"{bono_data.get('convexidad', 0):.2f}")
            dmod = bono_data.get("duracion_modificada", 0)
            st.info(f"Una duracion modificada de {dmod:.2f} implica que si las tasas suben 1%, el precio del bono caera aproximadamente un {dmod:.2f}%.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 10 — OPCIONES
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "🎲 Modulo 10 - Opciones":
    page_header("Modulo 10 · Opciones", "Valoracion Black-Scholes y las 5 Griegas - multiples activos")

    with st.sidebar:
        st.markdown("### Activos")
        tickers_m10 = build_ticker_selector("m10", default_all=False)
        st.markdown("---")
        st.markdown("### Parametros Opcion")
        strike = st.number_input("Strike (K)", value=150.0, key="m10_strike")
        dias = st.slider("Dias a Vencimiento", 1, 365, 30, key="m10_dias")
        tasa = st.number_input("Tasa Libre de Riesgo (%)", value=4.5, key="m10_tasa") / 100
        calcular = st.button("Valorar", type="primary", use_container_width=True)

    st.markdown(f'<div class="autoload-banner">Valorando opciones para <b>{len(tickers_m10)} activos</b>: {", ".join(tickers_m10[:6])}{"..." if len(tickers_m10) > 6 else ""}</div>', unsafe_allow_html=True)

    results_m10 = {}
    if calcular:
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, t in enumerate(tickers_m10):
            status_text.text(f"Calculando {t} ({i+1}/{len(tickers_m10)})...")
            payload = {"ticker": t, "strike": strike, "vencimiento_dias": dias, "tasa_libre_riesgo": tasa}
            data = api_post("/api/opciones/valorar", payload)
            if data:
                results_m10[t] = data
                st.session_state[f"m10_{t}"] = data
            progress_bar.progress((i + 1) / len(tickers_m10))
        progress_bar.empty()
        status_text.empty()
        if results_m10:
            st.success(f"Valoracion completada para {len(results_m10)} activos")

    if not results_m10:
        for t in tickers_m10:
            cached = st.session_state.get(f"m10_{t}")
            if cached:
                results_m10[t] = cached

    if results_m10:
        tabs = st.tabs([f"{t} · {TICKERS[t]}" for t in results_m10])
        for (ticker, bs_data), tab in zip(results_m10.items(), tabs):
            with tab:
                col_p1, col_p2, col_s = st.columns(3)
                col_p1.metric("Precio Call", f"${bs_data.get('call_price', 0):,.2f}")
                col_p2.metric("Precio Put", f"${bs_data.get('put_price', 0):,.2f}")
                col_s.metric("Precio Spot (S)", f"${bs_data.get('precio_spot', 0):,.2f}")
                st.markdown(f"**Volatilidad historica (sigma):** {bs_data.get('sigma', 0):.4f} · **T (anos):** {bs_data.get('T_anios', 0):.4f}")

                st.markdown("### Las 5 Griegas")
                df_greeks = pd.DataFrame({
                    "Griega": ["Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ)", "Rho (ρ)"],
                    "Valor": [
                        bs_data.get("delta_call", 0), bs_data.get("gamma", 0),
                        bs_data.get("vega", 0), bs_data.get("theta_call", 0), bs_data.get("rho_call", 0)
                    ],
                    "Significado": [
                        "Sensibilidad al precio", "Curvatura (velocidad)",
                        "Sensibilidad a volatilidad", "Decaimiento temporal", "Sensibilidad a tasa interes"
                    ]
                })
                st.dataframe(df_greeks.style.format({"Valor": "{:.4f}"}), use_container_width=True)
                st.caption("Theta y Rho suelen ser valores pequenos: cambio por dia o por 1% de tasa respectivamente.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULO 11 — STRESS TESTING
# ══════════════════════════════════════════════════════════════════════════════
elif opcion == "⚠️ Modulo 11 - Stress Testing":
    page_header("Modulo 11 · Stress Testing", "Simulacion de escenarios extremos y riesgo de cola")

    with st.sidebar:
        st.markdown("### Activos")
        tickers_st = build_ticker_selector("m11", default_all=False)
        st.markdown("---")
        st.markdown("### Parametros")
        inversion = st.number_input("Inversion Total", value=100000, step=10000, key="m11_inv")
        calcular = st.button("Ejecutar Stress", type="primary", use_container_width=True)

    if calcular and tickers_st:
        payload = {"tickers": tickers_st, "inversion": inversion, "confianza": 0.99}
        with st.spinner("Simulando escenarios de crisis..."):
            stress_data = api_post("/api/stress/calcular", payload)
        if stress_data:
            st.markdown("### Escenarios de Crisis")
            if "escenarios" in stress_data:
                df_escenarios = pd.DataFrame(stress_data["escenarios"])
                st.dataframe(df_escenarios.style.format({"perdida_usd": "${:,.2f}", "perdida_pct": "{:.2%}"}), use_container_width=True)

                if "perdida_usd" in df_escenarios.columns and "escenario" in df_escenarios.columns:
                    fig_stress = px.bar(df_escenarios, x="escenario", y="perdida_usd",
                        title="Perdida Estimada por Escenario",
                        color="perdida_usd", color_continuous_scale="Reds")
                    fig_stress.update_layout(**PLOT_TPL)
                    st.plotly_chart(fig_stress, use_container_width=True)

            badge_html("El Stress Testing revela la vulnerabilidad del portafolio ante eventos raros que el VaR parametrico podria subestimar.", "warning")
        else:
            st.error("Error al conectar con el modulo de Stress Testing.")
    elif not calcular:
        st.info("Selecciona activos y pulsa Ejecutar Stress.")
