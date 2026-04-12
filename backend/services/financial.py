"""
services/financial.py
Toda la lógica de cálculo financiero, extraída de los módulos Streamlit.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, jarque_bera, shapiro
import statsmodels.api as sm
from arch import arch_model
from functools import lru_cache
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

TICKERS_DEFAULT = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp.",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "GLD": "SPDR Gold ETF",
    "BTC-USD": "Bitcoin USD"
}

DIST_MAP = {
    "Normal": "normal",
    "t-Student": "t",
    "Skewed t-Student": "skewt"
}


# ─── UTILIDAD DE DESCARGA ─────────────────────────────────────────────────────

def get_data(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index()
    except Exception:
        return None


def calcular_indicadores(df: pd.DataFrame, sma_periodos=(20, 50),
                          rsi_periodo=14, bb_periodo=20, bb_std=2.0) -> pd.DataFrame:
    df = df.copy()
    for p in sma_periodos:
        df[f"SMA_{p}"] = df["Close"].rolling(p).mean()

    # EMA
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_periodo).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_periodo).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    rolling_mean = df["Close"].rolling(bb_periodo).mean()
    rolling_std = df["Close"].rolling(bb_periodo).std()
    df["BB_upper"] = rolling_mean + bb_std * rolling_std
    df["BB_lower"] = rolling_mean - bb_std * rolling_std
    df["BB_mid"]   = rolling_mean

    # Stochastic
    low_min  = df["Low"].rolling(14).min()
    high_max = df["High"].rolling(14).max()
    df["%K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["%D"] = df["%K"].rolling(3).mean()

    return df


# ─── SERVICIO 1: ANÁLISIS TÉCNICO ─────────────────────────────────────────────

def calcular_tecnico(ticker: str, periodo: str, sma_corto: int, sma_largo: int,
                     rsi_periodo: int, bb_periodo: int, bb_std: float) -> dict:
    df_raw = get_data(ticker, period=periodo)
    if df_raw is None or df_raw.empty:
        raise ValueError(f"No se encontraron datos para {ticker}")

    df = calcular_indicadores(
        df_raw,
        sma_periodos=[sma_corto, sma_largo],
        rsi_periodo=rsi_periodo,
        bb_periodo=bb_periodo,
        bb_std=bb_std
    )

    ultimo = df.iloc[-1]
    retorno = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

    def s(val):
        return None if pd.isna(val) else round(float(val), 4)

    datos = []
    for idx, row in df.tail(120).iterrows():
        datos.append({
            "fecha": str(idx.date()),
            "open": s(row["Open"]), "high": s(row["High"]),
            "low": s(row["Low"]),   "close": s(row["Close"]),
            f"sma_{sma_corto}": s(row.get(f"SMA_{sma_corto}")),
            f"sma_{sma_largo}": s(row.get(f"SMA_{sma_largo}")),
            "ema_20": s(row.get("EMA_20")),
            "rsi": s(row.get("RSI")),
            "macd": s(row.get("MACD")),
            "macd_signal": s(row.get("MACD_Signal")),
            "bb_upper": s(row.get("BB_upper")),
            "bb_lower": s(row.get("BB_lower")),
        })

    return {
        "ticker": ticker,
        "periodo": periodo,
        "ultimo_precio": round(float(ultimo["Close"]), 2),
        "retorno_periodo_pct": round(float(retorno), 2),
        "rsi_actual": round(float(ultimo.get("RSI", 0) or 0), 2),
        "volatilidad_diaria_pct": round(float(df["Close"].pct_change().std() * 100), 4),
        "sma_corto": s(ultimo.get(f"SMA_{sma_corto}")),
        "sma_largo": s(ultimo.get(f"SMA_{sma_largo}")),
        "bb_upper": s(ultimo.get("BB_upper")),
        "bb_lower": s(ultimo.get("BB_lower")),
        "datos": datos,
    }


# ─── SERVICIO 2: RENDIMIENTOS ──────────────────────────────────────────────────

def calcular_rendimientos(ticker: str, periodo: str, tipo: str) -> dict:
    df = get_data(ticker, period=periodo)
    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}")

    df["Rend_Simple"] = df["Close"].pct_change()
    df["Rend_Log"]    = np.log(df["Close"] / df["Close"].shift(1))
    col = "Rend_Simple" if tipo == "Simple" else "Rend_Log"
    rend = df[col].dropna()

    jb_stat, jb_p = jarque_bera(rend)
    rend_sample = rend.sample(min(len(rend), 5000), random_state=42)
    sh_stat, sh_p = shapiro(rend_sample)

    return {
        "ticker": ticker,
        "tipo": tipo,
        "media": round(float(rend.mean()), 6),
        "desviacion": round(float(rend.std()), 6),
        "asimetria": round(float(rend.skew()), 4),
        "curtosis": round(float(rend.kurtosis()), 4),
        "jarque_bera_pvalor": round(float(jb_p), 6),
        "shapiro_pvalor": round(float(sh_p), 6),
        "es_normal_jb": bool(jb_p >= 0.05),
        "es_normal_sw": bool(sh_p >= 0.05),
        "datos_rendimientos": [
            {"fecha": str(i.date()), "rendimiento": round(float(v), 6)}
            for i, v in rend.items()
        ],
    }


# ─── SERVICIO 3: GARCH ────────────────────────────────────────────────────────

def calcular_garch(ticker: str, horizonte: int, distribucion: str) -> dict:
    df = get_data(ticker, period="5y")
    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}")

    dist_final = DIST_MAP.get(distribucion, "t")
    returns = 100 * np.log(df["Close"] / df["Close"].shift(1)).dropna()

    specs = {
        "ARCH(1)":    {"p": 1, "q": 0, "vol": "GARCH"},
        "GARCH(1,1)": {"p": 1, "q": 1, "vol": "GARCH"},
        "EGARCH(1,1)":{"p": 1, "q": 1, "vol": "EGARCH"},
    }

    resultados = []
    res_garch = None
    for nombre, p in specs.items():
        try:
            mod = arch_model(returns, p=p["p"], q=p["q"], vol=p["vol"], dist=dist_final)
            res = mod.fit(disp="off")
            resultados.append({
                "modelo": nombre,
                "log_likelihood": round(float(res.loglikelihood), 4),
                "aic": round(float(res.aic), 4),
                "bic": round(float(res.bic), 4),
            })
            if nombre == "GARCH(1,1)":
                res_garch = res
        except Exception as e:
            resultados.append({"modelo": nombre, "error": str(e)})

    pronostico, jb_p, residuos = [], 0.0, []
    if res_garch is not None:
        fc = res_garch.forecast(horizon=horizonte, reindex=False)
        pronostico = [round(float(v), 4) for v in np.sqrt(fc.variance.values[-1])]
        std_resid = (res_garch.resid / res_garch.conditional_volatility).dropna()
        _, jb_p = jarque_bera(std_resid)
        residuos = [round(float(v), 4) for v in std_resid.tolist()]

    return {
        "ticker": ticker,
        "distribucion": distribucion,
        "comparativa_modelos": resultados,
        "pronostico_volatilidad": pronostico,
        "jb_residuos_pvalor": round(float(jb_p), 6),
        "residuos_std": residuos[-252:],   # último año
    }


# ─── SERVICIO 4: CAPM ─────────────────────────────────────────────────────────

def calcular_capm(ticker: str, benchmark: str, periodo: str) -> dict:
    df_rf = get_data("^TNX", period="5d")
    rf_anual = float(df_rf["Close"].iloc[-1]) / 100 if df_rf is not None and not df_rf.empty else 0.04

    df_asset = get_data(ticker, period=periodo)
    df_bench = get_data(benchmark, period=periodo)
    if df_asset is None or df_bench is None:
        raise ValueError("No se pudo descargar datos")

    for df in [df_asset, df_bench]:
        df.index = pd.to_datetime(df.index).tz_localize(None)

    asset_ret = df_asset["Close"].pct_change().dropna()
    bench_ret = df_bench["Close"].pct_change().dropna()

    data = pd.merge(asset_ret, bench_ret, left_index=True, right_index=True, how="inner").dropna()
    data.columns = ["Asset", "Market"]

    X = sm.add_constant(data["Market"])
    model = sm.OLS(data["Asset"], X).fit()
    beta = float(model.params["Market"])
    r2   = float(model.rsquared)

    rm_anual = float(data["Market"].mean()) * 252
    expected_return = rf_anual + beta * (rm_anual - rf_anual)

    if beta > 1.1:
        clasificacion = "Agresivo"
    elif beta < 0.9:
        clasificacion = "Defensivo"
    else:
        clasificacion = "Neutro"

    datos_reg = [
        {"market": round(float(r["Market"]), 6), "asset": round(float(r["Asset"]), 6)}
        for _, r in data.iterrows()
    ]

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "beta": round(beta, 4),
        "retorno_esperado_pct": round(expected_return, 6),
        "rf_anual_pct": round(rf_anual, 4),
        "rm_anual_pct": round(rm_anual, 4),
        "r_squared": round(r2, 4),
        "clasificacion": clasificacion,
        "datos_regresion": datos_reg,
    }


# ─── SERVICIO 5: VaR / CVaR ───────────────────────────────────────────────────

def calcular_var(ticker: str, confianza: float, inversion: float, n_sims: int) -> dict:
    df = get_data(ticker, period="2y")
    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}")

    returns = df["Close"].pct_change().dropna()
    mu, sigma = float(returns.mean()), float(returns.std())

    # Paramétrico
    var_param  = float(norm.ppf(1 - confianza, mu, sigma))
    var_anual  = var_param * np.sqrt(252)

    # Histórico
    var_hist   = float(np.percentile(returns, (1 - confianza) * 100))

    # Montecarlo
    sim = np.random.normal(mu, sigma, n_sims)
    var_mc     = float(np.percentile(sim, (1 - confianza) * 100))

    # CVaR
    cvar       = float(returns[returns <= var_hist].mean())

    return {
        "ticker": ticker,
        "confianza": confianza,
        "inversion": inversion,
        "var_parametrico_diario_pct": round(var_param, 6),
        "var_parametrico_anual_pct":  round(float(var_anual), 6),
        "var_historico_diario_pct":   round(var_hist, 6),
        "var_montecarlo_diario_pct":  round(var_mc, 6),
        "cvar_diario_pct":            round(cvar, 6),
        "perdida_param_usd":  round(abs(var_param * inversion), 2),
        "perdida_hist_usd":   round(abs(var_hist  * inversion), 2),
        "perdida_mc_usd":     round(abs(var_mc    * inversion), 2),
        "perdida_cvar_usd":   round(abs(cvar      * inversion), 2),
        "datos_rendimientos": [round(float(v), 6) for v in returns.tolist()],
    }


# ─── SERVICIO 6: MARKOWITZ ────────────────────────────────────────────────────

def calcular_markowitz(tickers: list[str], num_portafolios: int, periodo: str) -> dict:
    all_prices = {}
    for t in tickers:
        df_temp = get_data(t, period=periodo)
        if df_temp is not None:
            all_prices[t] = df_temp["Close"]

    if len(all_prices) < 2:
        raise ValueError("No se pudieron descargar suficientes activos")

    df_prices = pd.DataFrame(all_prices).dropna()
    returns   = np.log(df_prices / df_prices.shift(1)).dropna()

    corr = returns.corr().round(4)
    mean_ret = returns.mean() * 252
    cov_mat  = returns.cov() * 252
    n = len(all_prices)

    res = np.zeros((3, num_portafolios))
    weights_record = []
    rng = np.random.default_rng(42)

    for i in range(num_portafolios):
        w = rng.random(n)
        w /= w.sum()
        weights_record.append(w)
        p_ret = float(np.dot(w, mean_ret))
        p_std = float(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))))
        res[0, i] = p_std
        res[1, i] = p_ret
        res[2, i] = p_ret / p_std if p_std > 0 else 0

    df_sim = pd.DataFrame(res.T, columns=["vol", "ret", "sharpe"])
    max_idx = int(df_sim["sharpe"].idxmax())
    min_idx = int(df_sim["vol"].idxmin())

    def _port(idx, tipo):
        w = weights_record[idx]
        return {
            "tipo": tipo,
            "retorno_anual_pct": round(float(df_sim.iloc[idx]["ret"]), 4),
            "volatilidad_anual_pct": round(float(df_sim.iloc[idx]["vol"]), 4),
            "sharpe_ratio": round(float(df_sim.iloc[idx]["sharpe"]), 4),
            "pesos": {t: round(float(w[j]), 4) for j, t in enumerate(all_prices.keys())},
        }

    # Frontera eficiente (muestra 500 puntos)
    step = max(1, num_portafolios // 500)
    frontera = [
        {"volatilidad": round(float(df_sim.iloc[i]["vol"]), 4),
         "retorno": round(float(df_sim.iloc[i]["ret"]), 4),
         "sharpe": round(float(df_sim.iloc[i]["sharpe"]), 4)}
        for i in range(0, num_portafolios, step)
    ]

    return {
        "tickers": list(all_prices.keys()),
        "matriz_correlacion": corr.to_dict(),
        "frontera_eficiente": frontera,
        "portafolio_max_sharpe": _port(max_idx, "max_sharpe"),
        "portafolio_min_varianza": _port(min_idx, "min_varianza"),
    }


# ─── SERVICIO 7: SEÑALES ──────────────────────────────────────────────────────

def calcular_senales(ticker: str, rsi_up: int, rsi_down: int) -> dict:
    df = get_data(ticker, period="1y")
    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}")

    # Indicadores
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["MA20"]  = df["Close"].rolling(20).mean()
    df["STD20"] = df["Close"].rolling(20).std()
    df["BBU"]   = df["MA20"] + 2 * df["STD20"]
    df["BBL"]   = df["MA20"] - 2 * df["STD20"]

    curr, prev = df.iloc[-1], df.iloc[-2]
    senales = []

    # MACD
    if curr["MACD"] > curr["Signal_Line"] and prev["MACD"] <= prev["Signal_Line"]:
        senales.append({"indicador": "MACD", "estado": "COMPRA",  "descripcion": "Cruce alcista",  "color": "green"})
    elif curr["MACD"] < curr["Signal_Line"] and prev["MACD"] >= prev["Signal_Line"]:
        senales.append({"indicador": "MACD", "estado": "VENTA",   "descripcion": "Cruce bajista",  "color": "red"})
    else:
        senales.append({"indicador": "MACD", "estado": "NEUTRAL", "descripcion": "Sin cruces",     "color": "blue"})

    # RSI
    rsi_val = float(curr["RSI"])
    if rsi_val >= rsi_up:
        senales.append({"indicador": "RSI", "estado": "SOBRECOMPRA", "descripcion": f"Nivel: {rsi_val:.2f}", "color": "red"})
    elif rsi_val <= rsi_down:
        senales.append({"indicador": "RSI", "estado": "SOBREVENTA",  "descripcion": f"Nivel: {rsi_val:.2f}", "color": "green"})
    else:
        senales.append({"indicador": "RSI", "estado": "NEUTRAL",     "descripcion": f"Estable: {rsi_val:.2f}", "color": "blue"})

    # Bollinger
    if float(curr["Close"]) >= float(curr["BBU"]):
        senales.append({"indicador": "Bollinger", "estado": "VENTA",   "descripcion": "Precio sobre banda superior", "color": "red"})
    elif float(curr["Close"]) <= float(curr["BBL"]):
        senales.append({"indicador": "Bollinger", "estado": "COMPRA",  "descripcion": "Precio bajo banda inferior",  "color": "green"})
    else:
        senales.append({"indicador": "Bollinger", "estado": "NEUTRAL", "descripcion": "Dentro del canal",            "color": "blue"})

    # Medias
    if curr["SMA50"] > curr["SMA200"] and prev["SMA50"] <= prev["SMA200"]:
        senales.append({"indicador": "Medias", "estado": "GOLDEN CROSS", "descripcion": "Tendencia alcista fuerte", "color": "green"})
    elif curr["SMA50"] < curr["SMA200"] and prev["SMA50"] >= prev["SMA200"]:
        senales.append({"indicador": "Medias", "estado": "DEATH CROSS",  "descripcion": "Tendencia bajista fuerte", "color": "red"})
    else:
        est = "ALCISTA" if curr["SMA50"] > curr["SMA200"] else "BAJISTA"
        col = "green" if est == "ALCISTA" else "red"
        senales.append({"indicador": "Medias", "estado": est, "descripcion": "Sin cruces nuevos", "color": col})

    compras = sum(1 for s in senales if s["color"] == "green")
    ventas  = sum(1 for s in senales if s["color"] == "red")
    global_signal = "COMPRA" if compras > ventas else ("VENTA" if ventas > compras else "NEUTRAL")

    return {
        "ticker": ticker,
        "precio_actual": round(float(curr["Close"]), 2),
        "rsi_actual": round(rsi_val, 2),
        "senales": senales,
        "señal_global": global_signal,
    }


# ─── SERVICIO 8: MACRO ────────────────────────────────────────────────────────

def calcular_macro(tickers: list[str], benchmark: str, periodo: str) -> dict:
    all_data = {}
    for t in tickers:
        df_t = get_data(t, period=periodo)
        if df_t is not None:
            all_data[t] = df_t["Close"]

    df_bench = get_data(benchmark, period=periodo)
    if df_bench is None or len(all_data) == 0:
        raise ValueError("No se pudieron descargar datos")

    df = pd.DataFrame(all_data).dropna()
    weights = np.array([1 / len(df.columns)] * len(df.columns))

    port_ret  = df.pct_change().dropna()
    bench_ret = df_bench["Close"].pct_change().dropna()
    bench_ret = bench_ret.reindex(port_ret.index).dropna()
    port_ret  = port_ret.reindex(bench_ret.index).dropna()

    port_cum  = (1 + (port_ret @ weights)).cumprod() * 100
    bench_cum = (1 + bench_ret).cumprod() * 100

    rf = 0.04 / 252
    alpha           = float((port_cum.iloc[-1] - bench_cum.iloc[-1]) / 100)
    tracking_error  = float((port_ret @ weights - bench_ret).std() * np.sqrt(252))
    info_ratio      = alpha / tracking_error if tracking_error != 0 else 0.0

    peak    = port_cum.expanding().max()
    max_dd  = float(((port_cum - peak) / peak).min())
    vol_ann = float((port_ret @ weights).std() * np.sqrt(252))
    ret_ann = float(port_cum.iloc[-1] / 100 - 1)

    return {
        "rf_pct": 4.32,
        "rendimiento_portafolio_pct": round(ret_ann, 4),
        "rendimiento_benchmark_pct":  round(float(bench_cum.iloc[-1] / 100 - 1), 4),
        "alpha_pct":          round(alpha, 4),
        "tracking_error_pct": round(tracking_error, 4),
        "information_ratio":  round(info_ratio, 4),
        "max_drawdown_pct":   round(max_dd, 4),
        "volatilidad_anual_pct": round(vol_ann, 4),
        "portafolio_acumulado": [
            {"fecha": str(i.date()), "valor": round(float(v), 4)}
            for i, v in port_cum.items()
        ],
        "benchmark_acumulado": [
            {"fecha": str(i.date()), "valor": round(float(v), 4)}
            for i, v in bench_cum.items()
        ],
    }