"""
backend/services/financial.py
Lógica financiera completa para los 11 módulos (Fases 1-3).
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, jarque_bera, shapiro, chi2
import statsmodels.api as sm
from arch import arch_model
from scipy.optimize import curve_fit
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")

TICKERS_DEFAULT = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp.",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "GLD": "SPDR Gold ETF",
    "BTC-USD": "Bitcoin USD"
}
DIST_MAP = {"Normal": "normal", "t-Student": "t", "Skewed t-Student": "skewt"}

# ─── UTILIDAD DE DESCARGA ─────────────────────────────────────────────────────
def get_data(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index()
    except Exception:
        return None

# ═════════════════════════════════════════════════════════════════════════════
# MÓDULOS 1-8 (EXISTENTES)
# ═════════════════════════════════════════════════════════════════════════════
def calcular_indicadores(df, sma_periodos=(20, 50), rsi_periodo=14, bb_periodo=20, bb_std=2.0):
    df = df.copy()
    for p in sma_periodos: df[f"SMA_{p}"] = df["Close"].rolling(p).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_periodo).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_periodo).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    exp1, exp2 = df["Close"].ewm(span=12, adjust=False).mean(), df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"], df["MACD_Signal"] = exp1 - exp2, (exp1 - exp2).ewm(span=9, adjust=False).mean()
    rolling_mean, rolling_std = df["Close"].rolling(bb_periodo).mean(), df["Close"].rolling(bb_periodo).std()
    df["BB_upper"], df["BB_lower"] = rolling_mean + bb_std*rolling_std, rolling_mean - bb_std*rolling_std
    low_min, high_max = df["Low"].rolling(14).min(), df["High"].rolling(14).max()
    df["%K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["%D"] = df["%K"].rolling(3).mean()
    return df

def calcular_tecnico(ticker, periodo, sma_corto, sma_largo, rsi_periodo, bb_periodo, bb_std):
    df_raw = get_data(ticker, period=periodo)
    if df_raw is None or df_raw.empty: raise ValueError(f"Sin datos para {ticker}")
    df = calcular_indicadores(df_raw, sma_periodos=[sma_corto, sma_largo], rsi_periodo=rsi_periodo, bb_periodo=bb_periodo, bb_std=bb_std)
    ultimo = df.iloc[-1]
    retorno = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    def s(val): return None if pd.isna(val) else round(float(val), 4)
    datos = []
    for idx, row in df.tail(120).iterrows():
        datos.append({"fecha": str(idx.date()), "open": s(row["Open"]), "high": s(row["High"]), "low": s(row["Low"]), "close": s(row["Close"]),
                      f"sma_{sma_corto}": s(row.get(f"SMA_{sma_corto}")), f"sma_{sma_largo}": s(row.get(f"SMA_{sma_largo}")),
                      "ema_20": s(row.get("EMA_20")), "rsi": s(row.get("RSI")), "macd": s(row.get("MACD")),
                      "macd_signal": s(row.get("MACD_Signal")), "bb_upper": s(row.get("BB_upper")), "bb_lower": s(row.get("BB_lower"))})
    return {"ticker": ticker, "periodo": periodo, "ultimo_precio": round(float(ultimo["Close"]), 2),
            "retorno_periodo_pct": round(float(retorno), 2), "rsi_actual": round(float(ultimo.get("RSI", 0) or 0), 2),
            "volatilidad_diaria_pct": round(float(df["Close"].pct_change().std() * 100), 4),
            "sma_corto": s(ultimo.get(f"SMA_{sma_corto}")), "sma_largo": s(ultimo.get(f"SMA_{sma_largo}")),
            "bb_upper": s(ultimo.get("BB_upper")), "bb_lower": s(ultimo.get("BB_lower")), "datos": datos}

def calcular_rendimientos(ticker, periodo, tipo):
    df = get_data(ticker, period=periodo)
    if df is None or df.empty: raise ValueError(f"Sin datos para {ticker}")
    df["Rend_Simple"], df["Rend_Log"] = df["Close"].pct_change(), np.log(df["Close"] / df["Close"].shift(1))
    col = "Rend_Simple" if tipo == "Simple" else "Rend_Log"
    rend = df[col].dropna()
    jb_stat, jb_p = jarque_bera(rend)
    rend_sample = rend.sample(min(len(rend), 5000), random_state=42)
    sh_stat, sh_p = shapiro(rend_sample)
    return {"ticker": ticker, "tipo": tipo, "media": round(float(rend.mean()), 6), "desviacion": round(float(rend.std()), 6),
            "asimetria": round(float(rend.skew()), 4), "curtosis": round(float(rend.kurtosis()), 4),
            "jarque_bera_pvalor": round(float(jb_p), 6), "shapiro_pvalor": round(float(sh_p), 6),
            "es_normal_jb": bool(jb_p >= 0.05), "es_normal_sw": bool(sh_p >= 0.05),
            "datos_rendimientos": [{"fecha": str(i.date()), "rendimiento": round(float(v), 6)} for i, v in rend.items()]}

def calcular_garch(ticker, horizonte, distribucion):
    df = get_data(ticker, period="5y")
    if df is None or df.empty: raise ValueError(f"Sin datos para {ticker}")
    dist_final = DIST_MAP.get(distribucion, "t")
    returns = 100 * np.log(df["Close"] / df["Close"].shift(1)).dropna()
    specs = {"ARCH(1)": {"p": 1, "q": 0, "vol": "GARCH"}, "GARCH(1,1)": {"p": 1, "q": 1, "vol": "GARCH"}, "EGARCH(1,1)": {"p": 1, "q": 1, "vol": "EGARCH"}}
    resultados, res_garch = [], None
    for nombre, p in specs.items():
        try:
            mod = arch_model(returns, p=p["p"], q=p["q"], vol=p["vol"], dist=dist_final)
            res = mod.fit(disp="off")
            resultados.append({"modelo": nombre, "log_likelihood": round(float(res.loglikelihood), 4), "aic": round(float(res.aic), 4), "bic": round(float(res.bic), 4)})
            if nombre == "GARCH(1,1)": res_garch = res
        except Exception as e: resultados.append({"modelo": nombre, "error": str(e)})
    pronostico, jb_p, residuos = [], 0.0, []
    if res_garch is not None:
        fc = res_garch.forecast(horizon=horizonte, reindex=False)
        pronostico = [round(float(v), 4) for v in np.sqrt(fc.variance.values[-1])]
        std_resid = (res_garch.resid / res_garch.conditional_volatility).dropna()
        _, jb_p = jarque_bera(std_resid)
        residuos = [round(float(v), 4) for v in std_resid.tolist()]
    return {"ticker": ticker, "distribucion": distribucion, "comparativa_modelos": resultados,
            "pronostico_volatilidad": pronostico, "jb_residuos_pvalor": round(float(jb_p), 6), "residuos_std": residuos[-252:]}

def calcular_capm(ticker, benchmark, periodo):
    df_rf = get_data("^TNX", period="5d")
    rf_anual = float(df_rf["Close"].iloc[-1]) / 100 if df_rf is not None and not df_df.empty else 0.04
    df_asset, df_bench = get_data(ticker, period=periodo), get_data(benchmark, period=periodo)
    if df_asset is None or df_bench is None: raise ValueError("No se pudo descargar datos")
    for df in [df_asset, df_bench]: df.index = pd.to_datetime(df.index).tz_localize(None)
    asset_ret, bench_ret = df_asset["Close"].pct_change().dropna(), df_bench["Close"].pct_change().dropna()
    data = pd.merge(asset_ret, bench_ret, left_index=True, right_index=True, how="inner").dropna()
    data.columns = ["Asset", "Market"]
    model = sm.OLS(data["Asset"], sm.add_constant(data["Market"])).fit()
    beta, r2 = float(model.params["Market"]), float(model.rsquared)
    alpha_diario = float(model.params["const"])
    rm_anual = float(data["Market"].mean()) * 252
    expected_return = rf_anual + beta * (rm_anual - rf_anual)
    clasificacion = "Agresivo" if beta > 1.1 else ("Defensivo" if beta < 0.9 else "Neutro")
    datos_reg = [{"market": round(float(r["Market"]), 6), "asset": round(float(r["Asset"]), 6)} for _, r in data.iterrows()]
    return {"ticker": ticker, "benchmark": benchmark, "beta": round(beta, 4), "retorno_esperado_pct": round(expected_return, 6),
            "rf_anual_pct": round(rf_anual, 4), "rm_anual_pct": round(rm_anual, 4), "r_squared": round(r2, 4),
            "alpha_jensen_pct": round(alpha_diario * 252, 6), "clasificacion": clasificacion, "datos_regresion": datos_reg}

def calcular_var(ticker, confianza, inversion, n_sims):
    df = get_data(ticker, period="2y")
    if df is None or df.empty: raise ValueError(f"Sin datos para {ticker}")
    returns = df["Close"].pct_change().dropna()
    mu, sigma = float(returns.mean()), float(returns.std())
    var_param = float(norm.ppf(1 - confianza, mu, sigma))
    var_hist = float(np.percentile(returns, (1 - confianza) * 100))
    sim = np.random.normal(mu, sigma, n_sims)
    var_mc = float(np.percentile(sim, (1 - confianza) * 100))
    cvar = float(returns[returns <= var_hist].mean())
    # FASE 2.2: Test de Kupiec
    excedencias = int((returns < var_hist).sum())
    T = len(returns)
    p_hat = excedencias / T
    p_esperada = 1 - confianza
    if p_hat == 0 or p_hat == 1: kupiec_res = {"excedencias": excedencias, "p_valor": 1.0, "aprueba": True}
    else:
        LR_uc = -2 * (np.log((1-p_esperada)**(T-excedencias) * p_esperada**excedencias) - np.log((1-p_hat)**(T-excedencias) * p_hat**excedencias))
        p_valor = float(1 - chi2.cdf(LR_uc, df=1))
        kupiec_res = {"excedencias_obs": excedencias, "excedencias_esp": round(T * p_esperada, 1),
                      "LR_uc": round(float(LR_uc), 4), "p_valor_kupiec": round(p_valor, 4), "aprueba_kupiec": bool(p_valor > 0.05)}
    return {"ticker": ticker, "confianza": confianza, "inversion": inversion, "var_parametrico_diario_pct": round(var_param, 6),
            "var_historico_diario_pct": round(var_hist, 6), "var_montecarlo_diario_pct": round(var_mc, 6), "cvar_diario_pct": round(cvar, 6),
            "perdida_param_usd": round(abs(var_param * inversion), 2), "perdida_hist_usd": round(abs(var_hist * inversion), 2),
            "perdida_mc_usd": round(abs(var_mc * inversion), 2), "perdida_cvar_usd": round(abs(cvar * inversion), 2),
            "datos_rendimientos": [round(float(v), 6) for v in returns.tolist()], "kupiec": kupiec_res}

def calcular_markowitz(tickers, num_portafolios, periodo):
    all_prices = {}
    for t in tickers:
        df_temp = get_data(t, period=periodo)
        if df_temp is not None: all_prices[t] = df_temp["Close"]
    if len(all_prices) < 2: raise ValueError("No suficientes activos")
    df_prices = pd.DataFrame(all_prices).dropna()
    returns = np.log(df_prices / df_prices.shift(1)).dropna()
    corr = returns.corr().round(4)
    mean_ret = returns.mean() * 252
    cov_mat = returns.cov() * 252
    n = len(all_prices)
    # FASE 2.3: QP con cvxpy
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob_mv = cp.Problem(cp.Minimize(cp.quad_form(w, cov_mat.values)), constraints)
    prob_mv.solve(solver=cp.OSQP)
    w_mv = w.value
    y = cp.Variable(n)
    k = cp.Variable(nonneg=True)
    prob_ms = cp.Problem(cp.Minimize(cp.quad_form(y, cov_mat.values)), [mean_ret.values @ y - 0.04 * k == 1, cp.sum(y) == k, y >= 0])
    prob_ms.solve(solver=cp.OSQP)
    w_ms = y.value / k.value if k.value and k.value > 1e-8 else w_mv
    if w_mv is None: w_mv = np.ones(n)/n
    if w_ms is None: w_ms = np.ones(n)/n
    # Simulación MC para visualización
    res = np.zeros((3, num_portafolios))
    rng = np.random.default_rng(42)
    for i in range(num_portafolios):
        w = rng.random(n); w /= w.sum(); p_ret=float(np.dot(w, mean_ret)); p_std=float(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))))
        res[0, i], res[1, i], res[2, i] = p_std, p_ret, p_ret/p_std if p_std > 0 else 0
    df_sim = pd.DataFrame(res.T, columns=["vol", "ret", "sharpe"])
    def _port(w, tipo):
        ret = float(w @ mean_ret); risk = float(np.sqrt(w @ cov_mat @ w))
        return {"tipo": tipo, "retorno_anual_pct": round(ret, 4), "volatilidad_anual_pct": round(risk, 4),
                "sharpe_ratio": round((ret - 0.04) / risk, 4) if risk > 0 else 0,
                "pesos": {t: round(float(w[j]), 4) for j, t in enumerate(all_prices.keys())}}
    step = max(1, num_portafolios // 500)
    frontera = [{"volatilidad": round(float(df_sim.iloc[i]["vol"]), 4), "retorno": round(float(df_sim.iloc[i]["ret"]), 4),
                 "sharpe": round(float(df_sim.iloc[i]["sharpe"]), 4)} for i in range(0, num_portafolios, step)]
    return {"tickers": list(all_prices.keys()), "matriz_correlacion": corr.to_dict(), "frontera_eficiente": frontera,
            "portafolio_max_sharpe": _port(w_ms, "max_sharpe"), "portafolio_min_varianza": _port(w_mv, "min_varianza")}

def calcular_senales(ticker, rsi_up, rsi_down):
    df = get_data(ticker, period="1y")
    if df is None or df.empty: raise ValueError(f"Sin datos para {ticker}")
    df["SMA50"], df["SMA200"] = df["Close"].rolling(50).mean(), df["Close"].rolling(200).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    exp1, exp2 = df["Close"].ewm(span=12, adjust=False).mean(), df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"], df["Signal_Line"] = exp1 - exp2, (exp1 - exp2).ewm(span=9, adjust=False).mean()
    df["MA20"], df["STD20"] = df["Close"].rolling(20).mean(), df["Close"].rolling(20).std()
    df["BBU"], df["BBL"] = df["MA20"] + 2 * df["STD20"], df["MA20"] - 2 * df["STD20"]
    curr, prev = df.iloc[-1], df.iloc[-2]
    senales = []
    if curr["MACD"] > curr["Signal_Line"] and prev["MACD"] <= prev["Signal_Line"]: senales.append({"indicador": "MACD", "estado": "COMPRA", "descripcion": "Cruce alcista", "color": "green"})
    elif curr["MACD"] < curr["Signal_Line"] and prev["MACD"] >= prev["Signal_Line"]: senales.append({"indicador": "MACD", "estado": "VENTA", "descripcion": "Cruce bajista", "color": "red"})
    else: senales.append({"indicador": "MACD", "estado": "NEUTRAL", "descripcion": "Sin cruces", "color": "blue"})
    rsi_val = float(curr["RSI"])
    if rsi_val >= rsi_up: senales.append({"indicador": "RSI", "estado": "SOBRECOMPRA", "descripcion": f"Nivel: {rsi_val:.2f}", "color": "red"})
    elif rsi_val <= rsi_down: senales.append({"indicador": "RSI", "estado": "SOBREVENTA", "descripcion": f"Nivel: {rsi_val:.2f}", "color": "green"})
    else: senales.append({"indicador": "RSI", "estado": "NEUTRAL", "descripcion": f"Estable: {rsi_val:.2f}", "color": "blue"})
    if float(curr["Close"]) >= float(curr["BBU"]): senales.append({"indicador": "Bollinger", "estado": "VENTA", "descripcion": "Sobre banda superior", "color": "red"})
    elif float(curr["Close"]) <= float(curr["BBL"]): senales.append({"indicador": "Bollinger", "estado": "COMPRA", "descripcion": "Bajo banda inferior", "color": "green"})
    else: senales.append({"indicador": "Bollinger", "estado": "NEUTRAL", "descripcion": "Dentro del canal", "color": "blue"})
    if curr["SMA50"] > curr["SMA200"] and prev["SMA50"] <= prev["SMA200"]: senales.append({"indicador": "Medias", "estado": "GOLDEN CROSS", "descripcion": "Tendencia alcista fuerte", "color": "green"})
    elif curr["SMA50"] < curr["SMA200"] and prev["SMA50"] >= prev["SMA200"]: senales.append({"indicador": "Medias", "estado": "DEATH CROSS", "descripcion": "Tendencia bajista fuerte", "color": "red"})
    else:
        est = "ALCISTA" if curr["SMA50"] > curr["SMA200"] else "BAJISTA"
        col = "green" if est == "ALCISTA" else "red"
        senales.append({"indicador": "Medias", "estado": est, "descripcion": "Sin cruces nuevos", "color": col})
    low_min, high_max = df["Low"].rolling(14).min(), df["High"].rolling(14).max()
    df["%K"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["%D"] = df["%K"].rolling(3).mean()
    curr = df.iloc[-1]; prev = df.iloc[-2]
    k_curr, d_curr = float(curr["%K"]), float(curr["%D"])
    k_prev, d_prev = float(prev["%K"]), float(prev["%D"])
    if k_curr < 20 and k_prev <= d_prev and k_curr > d_curr: senales.append({"indicador": "Estocástico", "estado": "COMPRA", "descripcion": f"%K={k_curr:.1f} cruza %D en sobreventa", "color": "green"})
    elif k_curr > 80 and k_prev >= d_prev and k_curr < d_curr: senales.append({"indicador": "Estocástico", "estado": "VENTA", "descripcion": f"%K={k_curr:.1f} cruza %D en sobrecompra", "color": "red"})
    else:
        zona = "sobrecompra" if k_curr > 80 else "sobreventa" if k_curr < 20 else "zona media"
        senales.append({"indicador": "Estocástico", "estado": "NEUTRAL", "descripcion": f"%K={k_curr:.1f} · {zona}", "color": "blue"})
    compras = sum(1 for s in senales if s["color"] == "green")
    ventas = sum(1 for s in senales if s["color"] == "red")
    global_signal = "COMPRA" if compras > ventas else ("VENTA" if ventas > compras else "NEUTRAL")
    return {"ticker": ticker, "precio_actual": round(float(curr["Close"]), 2), "rsi_actual": round(rsi_val, 2), "senales": senales, "señal_global": global_signal}

def calcular_macro(tickers, benchmark, periodo):
    all_data = {}
    for t in tickers:
        df_t = get_data(t, period=periodo)
        if df_t is not None: all_data[t] = df_t["Close"]
    df_bench = get_data(benchmark, period=periodo)
    if df_bench is None or len(all_data) == 0: raise ValueError("No se pudieron descargar datos")
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
    return {"rf_pct": 4.32, "rendimiento_portafolio_pct": round(ret_ann, 4), "rendimiento_benchmark_pct": round(float(bench_cum.iloc[-1] / 100 - 1), 4),
            "alpha_pct": round(alpha, 4), "tracking_error_pct": round(tracking_error, 4), "information_ratio": round(info_ratio, 4),
            "max_drawdown_pct": round(max_dd, 4), "volatilidad_anual_pct": round(vol_ann, 4),
            "portafolio_acumulado": [{"fecha": str(i.date()), "valor": round(float(v), 4)} for i, v in port_cum.items()],
            "benchmark_acumulado": [{"fecha": str(i.date()), "valor": round(float(v), 4)} for i, v in bench_cum.items()]}

# ═════════════════════════════════════════════════════════════════════════════
# NUEVAS FUNCIONES FASE 2-3 (MÓDULOS 9-11)
# ═════════════════════════════════════════════════════════════════════════════

# FASE 2.1: EWMA de volatilidad (RiskMetrics λ=0.94)
def calcular_ewma(ticker: str, lambda_: float = 0.94, periodo: str = "2y") -> dict:
    df = get_data(ticker, period=periodo)
    if df is None or df.empty: raise ValueError(f"Sin datos para {ticker}")
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    var = np.zeros(len(returns))
    var[0] = returns.var()
    for i in range(1, len(returns)): var[i] = lambda_ * var[i-1] + (1 - lambda_) * returns.iloc[i-1]**2
    vol = np.sqrt(var) * np.sqrt(252) * 100
    return {"ticker": ticker, "lambda": lambda_, "volatilidad_ewma_anual_pct": round(float(vol[-1]), 4),
            "serie_vol": [round(float(v), 4) for v in vol[-252:]], "fechas": [str(d.date()) for d in returns.index[-252:]]}

# FASE 3.1: Curva de rendimiento Nelson-Siegel
def nelson_siegel_func(t, b0, b1, b2, lam):
    x = t / lam
    return b0 + b1 * ((1 - np.exp(-x)) / (x + 1e-6)) + b2 * (((1 - np.exp(-x)) / (x + 1e-6)) - np.exp(-x))

def calcular_curva_rendimiento() -> dict:
    tickers = ["^IRX", "^FVX", "^TNX", "^TYX"]
    plazos = [0.25, 5, 10, 30]
    tasas = []
    for t in tickers:
        try:
            df = get_data(t, period="1mo")
            tasas.append(float(df["Close"].iloc[-1]) / 100 if df is not None else 0.04)
        except: tasas.append(0.04)
    try:
        popt, _ = curve_fit(nelson_siegel_func, plazos, tasas, p0=[0.04, -0.02, 0.02, 2.0], maxfev=5000)
        beta0, beta1, beta2, lam = popt
        t_vals = np.linspace(0.25, 30, 100)
        y_vals = nelson_siegel_func(t_vals, beta0, beta1, beta2, lam)
        return {"beta0": float(beta0), "beta1": float(beta1), "beta2": float(beta2), "lambda": float(lam),
                "curva": [{"vencimiento": float(t), "tasa": float(y)} for t, y in zip(t_vals, y_vals)],
                "puntos": [{"vencimiento": p, "tasa": round(t, 4)} for p, t in zip(plazos, tasas)]}
    except Exception as e: return {"error": str(e), "puntos": [{"vencimiento": p, "tasa": round(t, 4)} for p, t in zip(plazos, tasas)]}

def calcular_bono(cupon_pct: float, vencimiento: int, valor_nominal: float, frecuencia: int) -> dict:
    ytm = 0.045
    n_periodos = vencimiento * frecuencia
    cupon_periodo = cupon_pct / 100 * valor_nominal / frecuencia
    ytm_periodo = ytm / frecuencia
    flujos = [cupon_periodo] * n_periodos
    flujos[-1] += valor_nominal
    precio = sum(cf / (1 + ytm_periodo)**i for i, cf in enumerate(flujos, 1))
    dur_mac = sum((i/frecuencia) * (cf / (1 + ytm_periodo)**i) for i, cf in enumerate(flujos, 1)) / precio
    dur_mod = dur_mac / (1 + ytm_periodo)
    convex = sum((i/frecuencia) * ((i/frecuencia) + 1/frecuencia) * (cf / (1 + ytm_periodo)**i) for i, cf in enumerate(flujos, 1)) / (precio * (1 + ytm_periodo)**2)
    return {"precio": round(precio, 4), "ytm_pct": round(ytm * 100, 4), "duracion_macaulay": round(dur_mac, 4),
            "duracion_modificada": round(dur_mod, 4), "convexidad": round(convex, 4), "cupon_pct": cupon_pct,
            "vencimiento": vencimiento, "valor_nominal": valor_nominal, "frecuencia": frecuencia}

# FASE 3.2: Black-Scholes + 5 Greeks
def calcular_opciones(ticker: str, strike: float, dias: int, tasa: float) -> dict:
    df = get_data(ticker, period="1y")
    if df is None or df.empty: raise ValueError(f"Sin datos para {ticker}")
    S = float(df["Close"].iloc[-1])
    sigma = float(df["Close"].pct_change().std() * np.sqrt(252))
    T = dias / 365
    if T <= 0 or sigma <= 0: raise ValueError("Parámetros inválidos")
    d1 = (np.log(S / strike) + (tasa + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - strike * np.exp(-tasa * T) * norm.cdf(d2)
    put_price = strike * np.exp(-tasa * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta_call = float(norm.cdf(d1))
    delta_put = float(norm.cdf(d1) - 1)
    gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    vega = float(S * np.sqrt(T) * norm.pdf(d1) / 100)
    theta_call = float((-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - tasa * strike * np.exp(-tasa * T) * norm.cdf(d2)) / 365)
    rho_call = float(strike * T * np.exp(-tasa * T) * norm.cdf(d2) / 100)
    return {"ticker": ticker, "precio_spot": round(S, 4), "strike": strike, "sigma": round(sigma, 4), "T_anios": round(T, 4),
            "call_price": round(float(call_price), 4), "put_price": round(float(put_price), 4),
            "delta_call": round(delta_call, 4), "delta_put": round(delta_put, 4),
            "gamma": round(gamma, 6), "vega": round(vega, 4), "theta_call": round(theta_call, 4), "rho_call": round(rho_call, 4)}

# FASE 3.3: Stress Testing
def calcular_stress(tickers: list[str], inversion: float, confianza: float) -> dict:
    all_prices = {}
    for t in tickers:
        df_t = get_data(t, period="2y")
        if df_t is not None: all_prices[t] = df_t["Close"]
    if not all_prices: raise ValueError("No se pudieron descargar datos")
    df = pd.DataFrame(all_prices).dropna()
    weights = np.array([1 / len(df.columns)] * len(df.columns))
    returns = df.pct_change().dropna()
    port_ret = returns @ weights
    mu = float(port_ret.mean())
    sigma = float(port_ret.std())
    var_base = float(norm.ppf(1 - confianza, mu, sigma))
    df_spy = get_data("^GSPC", period="2y")
    if df_spy is not None:
        spy_ret = df_spy["Close"].pct_change().dropna()
        spy_ret = spy_ret.reindex(port_ret.index).dropna()
        port_ret_aligned = port_ret.reindex(spy_ret.index).dropna()
        betas = []
        for t in tickers:
            try:
                asset_ret = returns[t].reindex(spy_ret.index).dropna()
                cov = np.cov(asset_ret, spy_ret)
                betas.append(cov[0, 1] / cov[1, 1])
            except: betas.append(1.0)
        beta_port = float(np.array(betas) @ weights)
    else:
        beta_port = 1.0
        betas = [1.0] * len(tickers)
    escenarios = [
        {"nombre": "Shock tasa +200pb", "shock_mkt": -beta_port * 0.02},
        {"nombre": "Caída mercado -20%", "shock_mkt": -0.20},
        {"nombre": "Caída mercado -30%", "shock_mkt": -0.30},
        {"nombre": "Shock vol × 2", "shock_mkt": -beta_port * 2 * sigma * np.sqrt(252)},
        {"nombre": "Tormenta perfecta", "shock_mkt": -0.20 - beta_port * 0.02}
    ]
    resultados = []
    heatmap = {}
    for esc in escenarios:
        perdida = esc["shock_mkt"]
        var_est = float(norm.ppf(1 - confianza, mu + perdida, sigma * (2 if "vol" in esc["nombre"] else 1)))
        resultados.append({"nombre": esc["nombre"], "perdida_pct": round(perdida, 4), "perdida_usd": round(abs(perdida) * inversion, 2), "var_estresado_pct": round(var_est, 4)})
        col = {}
        for t, b in zip(tickers, betas): col[t] = round(esc["shock_mkt"] * b, 4)
        heatmap[esc["nombre"]] = col
    reverse_shock = -0.20 / beta_port if beta_port != 0 else -0.20
    return {"tickers": tickers, "var_base_pct": round(var_base, 4), "var_base_usd": round(abs(var_base) * inversion, 2),
            "beta_portafolio": round(beta_port, 4), "escenarios": resultados, "reverse_stress_shock": round(reverse_shock, 4),
            "heatmap_activos": heatmap}