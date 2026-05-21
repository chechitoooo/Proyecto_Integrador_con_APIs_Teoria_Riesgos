"""
backend/services/financial.py
Logica financiera completa para los 11 modulos.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, jarque_bera, shapiro, chi2
import statsmodels.api as sm
from arch import arch_model
from scipy.optimize import curve_fit
import cvxpy as cp
import time as _time
import warnings
warnings.filterwarnings("ignore")

TICKERS_DEFAULT = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp.",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "GLD": "SPDR Gold ETF",
    "BTC-USD": "Bitcoin USD"
}
DIST_MAP = {"Normal": "normal", "t-Student": "t", "Skewed t-Student": "skewt"}

_data_cache = {}
_CACHE_TTL = 3600


# ─── DATA DOWNLOAD WITH CACHE ─────────────────────────────────────────────────

def get_data(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """Descarga datos de Yahoo Finance con cache transparente en memoria (TTL 1h)."""
    key = f"{ticker}_{period}"
    now = _time.time()
    if key in _data_cache:
        ts, cached_df = _data_cache[key]
        if now - ts < _CACHE_TTL:
            return cached_df.copy()
    try:
        df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        _data_cache[key] = (now, df.copy())
        return df
    except Exception:
        return None


def get_cache_info() -> dict:
    """Retorna info del cache para el endpoint de status."""
    return {
        "entradas": len(_data_cache),
        "ttl_segundos": _CACHE_TTL,
        "keys": list(_data_cache.keys()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 1 — INDICADORES TECNICOS
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_indicators(df, sma_periodos=(20, 50), rsi_periodo=14, bb_periodo=20, bb_std=2.0):
    df = df.copy()
    for p in sma_periodos:
        df[f"SMA_{p}"] = df["Close"].rolling(p).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_periodo).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_periodo).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    rolling_mean = df["Close"].rolling(bb_periodo).mean()
    rolling_std = df["Close"].rolling(bb_periodo).std()
    df["BB_upper"] = rolling_mean + bb_std * rolling_std
    df["BB_lower"] = rolling_mean - bb_std * rolling_std
    if "Low" in df.columns and "High" in df.columns:
        low_min = df["Low"].rolling(14).min()
        high_max = df["High"].rolling(14).max()
        df["%K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
        df["%D"] = df["%K"].rolling(3).mean()
    return df


def calcular_tecnico(ticker, periodo, sma_corto, sma_largo, rsi_periodo, bb_periodo, bb_std):
    df_raw = get_data(ticker, period=periodo)
    if df_raw is None or df_raw.empty:
        raise ValueError(f"Sin datos para {ticker}")
    df = _compute_indicators(df_raw, sma_periodos=[sma_corto, sma_largo],
                             rsi_periodo=rsi_periodo, bb_periodo=bb_periodo, bb_std=bb_std)
    ultimo = df.iloc[-1]
    retorno = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

    def s(val):
        return None if pd.isna(val) else round(float(val), 4)

    datos = []
    for idx, row in df.tail(120).iterrows():
        datos.append({
            "fecha": str(idx.date()),
            "open": s(row.get("Open")),
            "high": s(row.get("High")),
            "low": s(row.get("Low")),
            "close": s(row["Close"]),
            f"sma_{sma_corto}": s(row.get(f"SMA_{sma_corto}")),
            f"sma_{sma_largo}": s(row.get(f"SMA_{sma_largo}")),
            "ema": s(row.get("EMA_20")),
            "rsi": s(row.get("RSI")),
            "macd": s(row.get("MACD")),
            "macd_signal": s(row.get("MACD_Signal")),
            "bb_upper": s(row.get("BB_upper")),
            "bb_lower": s(row.get("BB_lower")),
            "stoch_k": s(row.get("%K")),
            "stoch_d": s(row.get("%D")),
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


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 2 — RENDIMIENTOS
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_rendimientos(ticker, periodo, tipo):
    df = get_data(ticker, period=periodo)
    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}")
    df["Rend_Simple"] = df["Close"].pct_change()
    df["Rend_Log"] = np.log(df["Close"] / df["Close"].shift(1))
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


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 3 — GARCH + EWMA
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_garch(ticker, horizonte, distribucion, lambda_ewma=0.94):
    df = get_data(ticker, period="5y")
    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}")
    dist_final = DIST_MAP.get(distribucion, "t")
    returns = 100 * np.log(df["Close"] / df["Close"].shift(1)).dropna()
    specs = {
        "ARCH(1)": {"p": 1, "q": 0, "vol": "GARCH"},
        "GARCH(1,1)": {"p": 1, "q": 1, "vol": "GARCH"},
        "EGARCH(1,1)": {"p": 1, "q": 1, "vol": "EGARCH"},
    }
    resultados, res_garch = [], None
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
    vol_condicional = []
    if res_garch is not None:
        fc = res_garch.forecast(horizon=horizonte, reindex=False)
        pronostico = [round(float(v), 4) for v in np.sqrt(fc.variance.values[-1])]
        std_resid = (res_garch.resid / res_garch.conditional_volatility).dropna()
        _, jb_p = jarque_bera(std_resid)
        residuos = [round(float(v), 4) for v in std_resid.tolist()]
        vol_condicional = [round(float(v), 4) for v in res_garch.conditional_volatility.dropna().tolist()[-252:]]

    # EWMA
    simple_ret = df["Close"].pct_change().dropna() * 100
    ewma_var = np.zeros(len(simple_ret))
    ewma_var[0] = float(simple_ret.var())
    for i in range(1, len(simple_ret)):
        ewma_var[i] = lambda_ewma * ewma_var[i-1] + (1 - lambda_ewma) * float(simple_ret.iloc[i-1])**2
    ewma_vol = np.sqrt(ewma_var)
    vol_rod = simple_ret.rolling(22).std().bfill().values
    fechas_vol = [str(d.date()) for d in simple_ret.index]

    # ARCH-LM
    arch_lm_p = 0.0
    if res_garch is not None and len(std_resid) > 10:
        resid_sq = std_resid**2
        lm_df = pd.DataFrame({"resid_sq": resid_sq})
        lm_df["lag"] = lm_df["resid_sq"].shift(1)
        lm_df = lm_df.dropna()
        if len(lm_df) > 10:
            try:
                lm_model = sm.OLS(lm_df["resid_sq"], sm.add_constant(lm_df["lag"])).fit()
                lm_stat = len(lm_df) * lm_model.rsquared
                arch_lm_p = float(1 - chi2.cdf(lm_stat, df=1))
            except Exception:
                arch_lm_p = 0.5

    return {
        "ticker": ticker,
        "distribucion": distribucion,
        "comparativa_modelos": resultados,
        "pronostico_volatilidad": pronostico,
        "volatilidad_condicional": vol_condicional,
        "jb_residuos_pvalor": round(float(jb_p), 6),
        "residuos_std": residuos[-252:] if len(residuos) > 252 else residuos,
        "ewma_volatilidad": [round(float(v), 4) for v in ewma_vol[-252:]] if len(ewma_vol) >= 252 else [round(float(v), 4) for v in ewma_vol],
        "vol_muestral_rodante": [round(float(v), 4) for v in vol_rod[-252:]],
        "fechas_vol": fechas_vol[-252:],
        "arch_lm_pvalor": round(float(arch_lm_p), 6),
    }


def calcular_ewma(ticker, lambda_, periodo):
    df = get_data(ticker, period=periodo)
    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}")
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    var = np.zeros(len(returns))
    var[0] = float(returns.var())
    for i in range(1, len(returns)):
        var[i] = lambda_ * var[i-1] + (1 - lambda_) * float(returns.iloc[i-1])**2
    vol = np.sqrt(var) * np.sqrt(252) * 100
    return {
        "ticker": ticker,
        "lambda": lambda_,
        "volatilidad_ewma_anual_pct": round(float(vol[-1]), 4),
        "serie_vol": [round(float(v), 4) for v in vol[-252:]],
        "fechas": [str(d.date()) for d in returns.index[-252:]],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 4 — CAPM
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_capm(ticker, benchmark, periodo):
    df_rf = get_data("^TNX", period="5d")
    rf_anual = float(df_rf["Close"].iloc[-1]) / 100 if (df_rf is not None and not df_rf.empty) else 0.04
    df_asset = get_data(ticker, period=periodo)
    df_bench = get_data(benchmark, period=periodo)
    if df_asset is None or df_bench is None:
        raise ValueError("No se pudo descargar datos del activo o benchmark")
    asset_ret = df_asset["Close"].pct_change().dropna()
    bench_ret = df_bench["Close"].pct_change().dropna()
    data = pd.merge(asset_ret, bench_ret, left_index=True, right_index=True, how="inner").dropna()
    data.columns = ["Asset", "Market"]
    model = sm.OLS(data["Asset"], sm.add_constant(data["Market"])).fit()
    beta = float(model.params["Market"])
    r2 = float(model.rsquared)
    alpha_diario = float(model.params["const"])
    rm_anual = float(data["Market"].mean()) * 252
    expected_return = rf_anual + beta * (rm_anual - rf_anual)
    clasificacion = "Agresivo" if beta > 1.1 else ("Defensivo" if beta < 0.9 else "Neutro")
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
        "alpha_jensen_pct": round(alpha_diario * 252, 6),
        "clasificacion": clasificacion,
        "datos_regresion": datos_reg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 5 — VaR + CVaR + Kupiec
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_kupiec(returns, var_val, confianza):
    excedencias = int((returns < var_val).sum())
    T = len(returns)
    p_esperada = 1 - confianza
    if T == 0:
        return {"excedencias": 0, "excedencias_esp": 0.0, "lr_uc": 0.0, "p_valor": 1.0, "aprueba": True}
    p_hat = max(min(excedencias / T, 0.999999), 0.000001)
    lr_uc = -2 * (
        np.log(((1 - p_esperada) ** (T - excedencias)) * (p_esperada ** excedencias))
        - np.log(((1 - p_hat) ** (T - excedencias)) * (p_hat ** excedencias))
    )
    p_valor = float(1 - chi2.cdf(lr_uc, df=1))
    return {
        "excedencias": excedencias,
        "excedencias_esp": round(T * p_esperada, 1),
        "lr_uc": round(float(lr_uc), 4),
        "p_valor": round(p_valor, 4),
        "aprueba": bool(p_valor > 0.05),
    }


def calcular_var(ticker, confianza, inversion, n_sims):
    df = get_data(ticker, period="2y")
    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}")
    returns = df["Close"].pct_change().dropna()
    mu = float(returns.mean())
    sigma = float(returns.std())

    var_param = float(norm.ppf(1 - confianza, mu, sigma))
    var_anual = var_param * np.sqrt(252)
    var_hist = float(np.percentile(returns, (1 - confianza) * 100))
    sim = np.random.default_rng(42).normal(mu, sigma, n_sims)
    var_mc = float(np.percentile(sim, (1 - confianza) * 100))
    cvar_series = returns[returns <= var_hist]
    cvar = float(cvar_series.mean()) if len(cvar_series) > 0 else float(var_hist)

    kupiec_param = _calc_kupiec(returns, var_param, confianza)
    kupiec_hist = _calc_kupiec(returns, var_hist, confianza)
    kupiec_mc = _calc_kupiec(returns, var_mc, confianza)

    return {
        "ticker": ticker, "confianza": confianza, "inversion": inversion,
        "var_parametrico_diario_pct": round(var_param, 6),
        "var_parametrico_anual_pct": round(var_anual, 6),
        "var_historico_diario_pct": round(var_hist, 6),
        "var_montecarlo_diario_pct": round(var_mc, 6),
        "cvar_diario_pct": round(cvar, 6),
        "perdida_param_usd": round(abs(var_param * inversion), 2),
        "perdida_hist_usd": round(abs(var_hist * inversion), 2),
        "perdida_mc_usd": round(abs(var_mc * inversion), 2),
        "perdida_cvar_usd": round(abs(cvar * inversion), 2),
        "datos_rendimientos": [round(float(v), 6) for v in returns.tolist()],
        "kupiec_parametrico": kupiec_param,
        "kupiec_historico": kupiec_hist,
        "kupiec_montecarlo": kupiec_mc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 6 — MARKOWITZ (QP + Montecarlo)
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_markowitz(tickers, num_portafolios, periodo, no_short_selling=True):
    all_prices = {}
    for t in tickers:
        df_temp = get_data(t, period=periodo)
        if df_temp is not None:
            all_prices[t] = df_temp["Close"]
    if len(all_prices) < 2:
        raise ValueError("No suficientes activos con datos")
    df_prices = pd.DataFrame(all_prices).dropna()
    returns = np.log(df_prices / df_prices.shift(1)).dropna()
    corr = returns.corr().round(4)
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    n = len(all_prices)

    # QP - min varianza
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1]
    if no_short_selling:
        constraints.append(w >= 0)
    prob_mv = cp.Problem(cp.Minimize(cp.quad_form(w, cov_mat.values)), constraints)
    prob_mv.solve(solver=cp.OSQP)
    w_mv = w.value if w.value is not None else np.ones(n) / n

    # QP - max Sharpe (via transformation)
    y = cp.Variable(n)
    k = cp.Variable(nonneg=True)
    ms_constraints = [mean_ret.values @ y - 0.04 * k == 1, cp.sum(y) == k]
    if no_short_selling:
        ms_constraints.append(y >= 0)
    prob_ms = cp.Problem(cp.Minimize(cp.quad_form(y, cov_mat.values)), ms_constraints)
    prob_ms.solve(solver=cp.OSQP)
    if y.value is not None and k.value is not None and k.value > 1e-8:
        w_ms = y.value / k.value
    else:
        w_ms = w_mv

    # Montecarlo para visualizacion
    res = np.zeros((3, num_portafolios))
    rng = np.random.default_rng(42)
    for i in range(num_portafolios):
        w_rand = rng.random(n)
        w_rand /= w_rand.sum()
        p_ret = float(np.dot(w_rand, mean_ret))
        p_std = float(np.sqrt(w_rand @ cov_mat.values @ w_rand))
        res[0, i] = p_std
        res[1, i] = p_ret
        res[2, i] = (p_ret - 0.04) / p_std if p_std > 0 else 0

    def _port(w_opt, tipo_str):
        ret = float(w_opt @ mean_ret)
        risk = float(np.sqrt(w_opt @ cov_mat.values @ w_opt))
        return {
            "tipo": tipo_str,
            "retorno_anual_pct": round(ret, 4),
            "volatilidad_anual_pct": round(risk, 4),
            "sharpe_ratio": round((ret - 0.04) / risk, 4) if risk > 0 else 0,
            "pesos": {t: round(float(w_opt[j]), 4) for j, t in enumerate(all_prices.keys())},
        }

    step = max(1, num_portafolios // 500)
    frontera = [
        {"volatilidad": round(float(res[0, i]), 4), "retorno": round(float(res[1, i]), 4),
         "sharpe": round(float(res[2, i]), 4)}
        for i in range(0, num_portafolios, step)
    ]
    return {
        "tickers": list(all_prices.keys()),
        "matriz_correlacion": corr.to_dict(),
        "frontera_eficiente": frontera,
        "portafolio_max_sharpe": _port(w_ms, "max_sharpe"),
        "portafolio_min_varianza": _port(w_mv, "min_varianza"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 7 — SENALES
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_senales(ticker, rsi_up, rsi_down):
    df = get_data(ticker, period="1y")
    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}")
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["STD20"] = df["Close"].rolling(20).std()
    df["BBU"] = df["MA20"] + 2 * df["STD20"]
    df["BBL"] = df["MA20"] - 2 * df["STD20"]
    if "Low" in df.columns and "High" in df.columns:
        low_min = df["Low"].rolling(14).min()
        high_max = df["High"].rolling(14).max()
        df["%K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
        df["%D"] = df["%K"].rolling(3).mean()
    else:
        df["%K"] = 50.0
        df["%D"] = 50.0

    curr, prev = df.iloc[-1], df.iloc[-2]
    senales = []

    # MACD
    if curr["MACD"] > curr["Signal_Line"] and prev["MACD"] <= prev["Signal_Line"]:
        senales.append({"indicador": "MACD", "estado": "COMPRA", "descripcion": "Cruce alcista", "color": "green"})
    elif curr["MACD"] < curr["Signal_Line"] and prev["MACD"] >= prev["Signal_Line"]:
        senales.append({"indicador": "MACD", "estado": "VENTA", "descripcion": "Cruce bajista", "color": "red"})
    else:
        senales.append({"indicador": "MACD", "estado": "NEUTRAL", "descripcion": "Sin cruces", "color": "blue"})

    rsi_val = float(curr["RSI"])
    if rsi_val >= rsi_up:
        senales.append({"indicador": "RSI", "estado": "SOBRECOMPRA", "descripcion": f"Nivel: {rsi_val:.2f}", "color": "red"})
    elif rsi_val <= rsi_down:
        senales.append({"indicador": "RSI", "estado": "SOBREVENTA", "descripcion": f"Nivel: {rsi_val:.2f}", "color": "green"})
    else:
        senales.append({"indicador": "RSI", "estado": "NEUTRAL", "descripcion": f"Estable: {rsi_val:.2f}", "color": "blue"})

    # Bollinger
    if float(curr["Close"]) >= float(curr["BBU"]):
        senales.append({"indicador": "Bollinger", "estado": "VENTA", "descripcion": "Sobre banda superior", "color": "red"})
    elif float(curr["Close"]) <= float(curr["BBL"]):
        senales.append({"indicador": "Bollinger", "estado": "COMPRA", "descripcion": "Bajo banda inferior", "color": "green"})
    else:
        senales.append({"indicador": "Bollinger", "estado": "NEUTRAL", "descripcion": "Dentro del canal", "color": "blue"})

    # Medias moviles
    if curr["SMA50"] > curr["SMA200"] and prev["SMA50"] <= prev["SMA200"]:
        senales.append({"indicador": "Medias", "estado": "GOLDEN CROSS", "descripcion": "Tendencia alcista fuerte", "color": "green"})
    elif curr["SMA50"] < curr["SMA200"] and prev["SMA50"] >= prev["SMA200"]:
        senales.append({"indicador": "Medias", "estado": "DEATH CROSS", "descripcion": "Tendencia bajista fuerte", "color": "red"})
    else:
        est = "ALCISTA" if curr["SMA50"] > curr["SMA200"] else "BAJISTA"
        col = "green" if est == "ALCISTA" else "red"
        senales.append({"indicador": "Medias", "estado": est, "descripcion": "Sin cruces nuevos", "color": col})

    k_curr, d_curr = float(curr["%K"]), float(curr["%D"])
    k_prev, d_prev = float(prev["%K"]), float(prev["%D"])
    if k_curr < 20 and k_prev <= d_prev and k_curr > d_curr:
        senales.append({"indicador": "Estocastico", "estado": "COMPRA", "descripcion": f"%K={k_curr:.1f} cruza %D en sobreventa", "color": "green"})
    elif k_curr > 80 and k_prev >= d_prev and k_curr < d_curr:
        senales.append({"indicador": "Estocastico", "estado": "VENTA", "descripcion": f"%K={k_curr:.1f} cruza %D en sobrecompra", "color": "red"})
    else:
        zona = "sobrecompra" if k_curr > 80 else "sobreventa" if k_curr < 20 else "zona media"
        senales.append({"indicador": "Estocastico", "estado": "NEUTRAL", "descripcion": f"%K={k_curr:.1f} - {zona}", "color": "blue"})

    compras = sum(1 for s in senales if s["color"] == "green")
    ventas = sum(1 for s in senales if s["color"] == "red")
    global_signal = "COMPRA" if compras > ventas else ("VENTA" if ventas > compras else "NEUTRAL")

    return {
        "ticker": ticker,
        "precio_actual": round(float(curr["Close"]), 2),
        "rsi_actual": round(rsi_val, 2),
        "senales": senales,
        "senal_global": global_signal,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 8 — MACRO + BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_macro(tickers, benchmark, periodo):
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
    port_ret = df.pct_change().dropna()
    bench_ret = df_bench["Close"].pct_change().dropna().reindex(port_ret.index).dropna()
    port_ret = port_ret.reindex(bench_ret.index).dropna()
    port_cum = (1 + (port_ret @ weights)).cumprod() * 100
    bench_cum = (1 + bench_ret).cumprod() * 100
    alpha = float((port_cum.iloc[-1] - bench_cum.iloc[-1]) / 100)
    tracking_error = float((port_ret @ weights - bench_ret).std() * np.sqrt(252))
    info_ratio = alpha / tracking_error if tracking_error != 0 else 0.0
    max_dd = float(((port_cum - port_cum.expanding().max()) / port_cum.expanding().max()).min())
    vol_ann = float((port_ret @ weights).std() * np.sqrt(252))
    ret_ann = float(port_cum.iloc[-1] / 100 - 1)
    return {
        "rf_pct": 4.32,
        "rendimiento_portafolio_pct": round(ret_ann, 4),
        "rendimiento_benchmark_pct": round(float(bench_cum.iloc[-1] / 100 - 1), 4),
        "alpha_pct": round(alpha, 4),
        "tracking_error_pct": round(tracking_error, 4),
        "information_ratio": round(info_ratio, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "volatilidad_anual_pct": round(vol_ann, 4),
        "portafolio_acumulado": [{"fecha": str(i.date()), "valor": round(float(v), 4)} for i, v in port_cum.items()],
        "benchmark_acumulado": [{"fecha": str(i.date()), "valor": round(float(v), 4)} for i, v in bench_cum.items()],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 9 — RENTA FIJA: Nelson-Siegel + Duracion/Convexidad
# ═══════════════════════════════════════════════════════════════════════════════

def nelson_siegel_func(t, b0, b1, b2, lam):
    x = t / lam
    term = (1 - np.exp(-x)) / (x + 1e-12)
    return b0 + b1 * term + b2 * (term - np.exp(-x))


def calcular_curva_rendimiento():
    tickers_map = {
        "^IRX": 0.25, "DGS1": 1, "DGS2": 2, "DGS3": 3,
        "^FVX": 5, "DGS7": 7, "DGS10": 10, "^TNX": 10,
        "DGS20": 20, "^TYX": 30, "DGS30": 30,
    }
    plazos = []
    tasas = []
    seen = set()
    for t, p in tickers_map.items():
        if p in seen:
            continue
        seen.add(p)
        try:
            df = get_data(t, period="1mo")
            val = float(df["Close"].iloc[-1]) / 100 if df is not None and not df.empty else None
            if val is None:
                continue
            plazos.append(p)
            tasas.append(val)
        except Exception:
            continue

    if len(plazos) < 4:
        plazos = [0.25, 5, 10, 30]
        tasas = [0.04, 0.042, 0.045, 0.048]

    try:
        popt, _ = curve_fit(nelson_siegel_func, plazos, tasas, p0=[0.04, -0.02, 0.02, 2.0], maxfev=5000)
        beta0, beta1, beta2, lam = popt
        fitted = np.array([nelson_siegel_func(t, beta0, beta1, beta2, lam) for t in plazos])
        rmse = float(np.sqrt(np.mean((np.array(tasas) - fitted) ** 2)))
        t_vals = np.linspace(0.25, 30, 100)
        y_vals = nelson_siegel_func(t_vals, beta0, beta1, beta2, lam)
        return {
            "beta0": float(beta0), "beta1": float(beta1), "beta2": float(beta2), "lambda": float(lam),
            "rmse": round(rmse, 6),
            "curva": [{"vencimiento": float(t), "tasa": float(y)} for t, y in zip(t_vals, y_vals)],
            "puntos": [{"vencimiento": p, "tasa": round(t, 4)} for p, t in zip(plazos, tasas)],
            "n_puntos": len(plazos),
        }
    except Exception as e:
        return {"error": str(e), "puntos": [{"vencimiento": p, "tasa": round(t, 4)} for p, t in zip(plazos, tasas)]}


def _bond_sensitivity(cupon_pct, vencimiento, valor_nominal, frecuencia):
    ytm = 0.045
    n_periodos = vencimiento * frecuencia
    cupon = cupon_pct / 100 * valor_nominal / frecuencia
    ytm_per = ytm / frecuencia
    flujos = [cupon] * n_periodos
    flujos[-1] += valor_nominal

    def _price(ytm_val):
        yp = ytm_val / frecuencia
        return sum(cf / (1 + yp) ** i for i, cf in enumerate(flujos, 1))

    precio_base = _price(ytm)
    dur_mac = sum((i / frecuencia) * (cf / (1 + ytm_per) ** i) for i, cf in enumerate(flujos, 1)) / precio_base
    dur_mod = dur_mac / (1 + ytm_per)
    convex = sum(
        (i / frecuencia) * ((i / frecuencia) + 1 / frecuencia) * (cf / (1 + ytm_per) ** i)
        for i, cf in enumerate(flujos, 1)
    ) / (precio_base * (1 + ytm_per) ** 2)

    sensibilidad = []
    for shock_bp in [50, 100, 200]:
        shock = shock_bp / 10000.0
        ytm_nuevo = ytm + shock
        precio_real = _price(ytm_nuevo)
        cambio_real = (precio_real - precio_base) / precio_base * 100
        aprox_linear = -dur_mod * shock * 100
        aprox_cuad = (-dur_mod * shock + 0.5 * convex * shock ** 2) * 100
        sensibilidad.append({
            "shock_bp": shock_bp,
            "precio_real": round(precio_real, 4),
            "cambio_real_pct": round(cambio_real, 4),
            "aprox_duracion_pct": round(aprox_linear, 4),
            "aprox_duracion_convexidad_pct": round(aprox_cuad, 4),
        })

    return {
        "precio": round(precio_base, 4), "ytm_pct": round(ytm * 100, 4),
        "duracion_macaulay": round(dur_mac, 4), "duracion_modificada": round(dur_mod, 4),
        "convexidad": round(convex, 4), "cupon_pct": cupon_pct,
        "vencimiento": vencimiento, "valor_nominal": valor_nominal, "frecuencia": frecuencia,
        "sensibilidad": sensibilidad,
    }


def calcular_bono(cupon_pct, vencimiento, valor_nominal, frecuencia):
    return _bond_sensitivity(cupon_pct, vencimiento, valor_nominal, frecuencia)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 10 — OPCIONES Black-Scholes + 5 Greeks
# ═══════════════════════════════════════════════════════════════════════════════

def _bs_price(S, K, T, r, sigma, tipo="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _implied_vol(S, K, T, r, market_price, tipo="call", max_iter=100, tol=1e-6):
    sigma = 0.3
    for _ in range(max_iter):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        if abs(vega) < 1e-12:
            break
        price = _bs_price(S, K, T, r, sigma, tipo)
        diff = price - market_price
        if abs(diff) < tol:
            return float(sigma)
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.01
    return float(sigma)


def calcular_opciones(ticker, strike, vencimiento_dias, tasa_libre_riesgo):
    df = get_data(ticker, period="1y")
    if df is None or df.empty:
        raise ValueError(f"Sin datos para {ticker}")
    S = float(df["Close"].iloc[-1])
    sigma_hist = float(df["Close"].pct_change().std() * np.sqrt(252))
    T = vencimiento_dias / 365.0
    if T <= 0 or sigma_hist <= 0:
        raise ValueError("Parametros invalidos: T o sigma <= 0")
    d1 = (np.log(S / strike) + (tasa_libre_riesgo + 0.5 * sigma_hist ** 2) * T) / (sigma_hist * np.sqrt(T))
    d2 = d1 - sigma_hist * np.sqrt(T)
    call_price = _bs_price(S, strike, T, tasa_libre_riesgo, sigma_hist, "call")
    put_price = _bs_price(S, strike, T, tasa_libre_riesgo, sigma_hist, "put")
    delta_call = float(norm.cdf(d1))
    delta_put = float(norm.cdf(d1) - 1)
    gamma = float(norm.pdf(d1) / (S * sigma_hist * np.sqrt(T)))
    vega = float(S * np.sqrt(T) * norm.pdf(d1) / 100)
    theta_call = float((-(S * norm.pdf(d1) * sigma_hist) / (2 * np.sqrt(T)) - tasa_libre_riesgo * strike * np.exp(-tasa_libre_riesgo * T) * norm.cdf(d2)) / 365)
    rho_call = float(strike * T * np.exp(-tasa_libre_riesgo * T) * norm.cdf(d2) / 100)

    # Put-call parity: C - P = S - K * exp(-rT)
    parity_lhs = call_price - put_price
    parity_rhs = S - strike * np.exp(-tasa_libre_riesgo * T)
    parity_error = abs(parity_lhs - parity_rhs)

    # Implied volatility from call price (Newton-Raphson)
    sigma_imp = _implied_vol(S, strike, T, tasa_libre_riesgo, call_price, "call")

    # Payoff and price curves for visualization
    S_range = np.linspace(S * 0.5, S * 1.5, 100)
    payoff_call = [max(s - strike, 0) for s in S_range]
    payoff_put = [max(strike - s, 0) for s in S_range]
    price_call_curve = [_bs_price(s, strike, T, tasa_libre_riesgo, sigma_hist, "call") for s in S_range]
    price_put_curve = [_bs_price(s, strike, T, tasa_libre_riesgo, sigma_hist, "put") for s in S_range]

    # Delta vs spot for different T
    delta_curves = []
    for mult in [0.5, 1.0, 2.0]:
        T_var = T * mult
        if T_var < 1 / 365:
            continue
        d1_var = (np.log(S_range / strike) + (tasa_libre_riesgo + 0.5 * sigma_hist ** 2) * T_var) / (sigma_hist * np.sqrt(T_var))
        delta_var = norm.cdf(d1_var)
        delta_curves.append({
            "T_anios": round(T_var, 4),
            "spot": [round(float(s), 2) for s in S_range],
            "delta": [round(float(d), 4) for d in delta_var],
        })

    return {
        "ticker": ticker, "precio_spot": round(S, 4), "strike": strike,
        "sigma": round(sigma_hist, 4), "T_anios": round(T, 4),
        "call_price": round(float(call_price), 4), "put_price": round(float(put_price), 4),
        "delta_call": round(delta_call, 4), "delta_put": round(delta_put, 4),
        "gamma": round(gamma, 6), "vega": round(vega, 4),
        "theta_call": round(theta_call, 4), "rho_call": round(rho_call, 4),
        "paridad_put_call": {"lhs": round(float(parity_lhs), 6), "rhs": round(float(parity_rhs), 6), "error": round(float(parity_error), 10)},
        "volatilidad_implicita": round(sigma_imp, 4),
        "curva_payoff": {
            "spot": [round(float(s), 2) for s in S_range],
            "payoff_call": [round(float(v), 2) for v in payoff_call],
            "payoff_put": [round(float(v), 2) for v in payoff_put],
            "precio_call": [round(float(v), 4) for v in price_call_curve],
            "precio_put": [round(float(v), 4) for v in price_put_curve],
        },
        "curvas_delta": delta_curves,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULO 11 — STRESS TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_stress(tickers, inversion, confianza):
    all_prices = {}
    for t in tickers:
        df_t = get_data(t, period="2y")
        if df_t is not None:
            all_prices[t] = df_t["Close"]
    if not all_prices:
        raise ValueError("No se pudieron descargar datos")
    df = pd.DataFrame(all_prices).dropna()
    weights = np.array([1 / len(df.columns)] * len(df.columns))
    returns = df.pct_change().dropna()
    port_ret = returns @ weights
    mu = float(port_ret.mean())
    sigma = float(port_ret.std())
    var_base = float(norm.ppf(1 - confianza, mu, sigma))

    df_spy = get_data("^GSPC", period="2y")
    betas = [1.0] * len(tickers)
    beta_port = 1.0
    if df_spy is not None:
        spy_ret = df_spy["Close"].pct_change().dropna()
        spy_ret = spy_ret.reindex(port_ret.index).dropna()
        port_ret_aligned = port_ret.reindex(spy_ret.index).dropna()
        betas = []
        for t in tickers:
            try:
                asset_ret = returns[t].reindex(spy_ret.index).dropna()
                cov = np.cov(asset_ret, spy_ret)
                betas.append(float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0)
            except Exception:
                betas.append(1.0)
        beta_port = float(np.array(betas) @ weights)

    escenarios = [
        {"nombre": "Shock tasa +200pb", "shock_mkt": -beta_port * 0.02},
        {"nombre": "Caida mercado -20%", "shock_mkt": -0.20},
        {"nombre": "Caida mercado -30%", "shock_mkt": -0.30},
        {"nombre": "Shock vol x 2", "shock_mkt": -beta_port * 2 * sigma * np.sqrt(252)},
        {"nombre": "Tormenta perfecta", "shock_mkt": -0.20 - beta_port * 0.02},
    ]
    resultados = []
    heatmap = {}
    for esc in escenarios:
        perdida = esc["shock_mkt"]
        var_est = float(norm.ppf(1 - confianza, mu + perdida, sigma * (2 if "vol" in esc["nombre"] else 1)))
        resultados.append({
            "nombre": esc["nombre"],
            "perdida_pct": round(perdida, 4),
            "perdida_usd": round(abs(perdida) * inversion, 2),
            "var_estresado_pct": round(var_est, 4),
        })
        heatmap[esc["nombre"]] = {t: round(esc["shock_mkt"] * b, 4) for t, b in zip(tickers, betas)}

    reverse_shock = -0.20 / beta_port if beta_port != 0 else -0.20
    return {
        "tickers": tickers,
        "var_base_pct": round(var_base, 4),
        "var_base_usd": round(abs(var_base) * inversion, 2),
        "beta_portafolio": round(beta_port, 4),
        "escenarios": resultados,
        "reverse_stress_shock": round(reverse_shock, 4),
        "heatmap_activos": heatmap,
        "escenario": [r["nombre"] for r in resultados],
        "perdida_usd": [r["perdida_usd"] for r in resultados],
        "perdida_pct": [r["perdida_pct"] for r in resultados],
    }
