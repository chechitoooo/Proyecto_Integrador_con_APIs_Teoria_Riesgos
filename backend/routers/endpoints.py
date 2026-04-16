"""
routers/endpoints.py
Los 8 routers temáticos (uno por módulo) + router de utilidades.
Total: 9 endpoints principales (supera el mínimo de 7 requerido).
"""
from fastapi import APIRouter, Depends, Query
from typing import Annotated
from typing import Annotated
from fastapi import Depends

from backend.models.schemas import (
    TecnicoRequest, RendimientosRequest, GarchRequest, CapmRequest,
    VarRequest, MarkowitzRequest, SenalesRequest, MacroRequest,
    IndicadoresResponse, RendimientosResponse, GarchResponse,
    CapmResponse, VarResponse, MarkowitzResponse, SenalesResponse, MacroResponse,
)
from .dependencies import FinancialService, get_financial_service
from backend.services.financial import TICKERS_DEFAULT

ServiceDep = Annotated[FinancialService, Depends(get_financial_service)]


# ─── ROUTER UTILIDADES ────────────────────────────────────────────────────────
router_utils = APIRouter(prefix="/api/utils", tags=["Utilidades"])

@router_utils.get("/tickers", summary="Lista de tickers disponibles")
def get_tickers():
    """Retorna el listado de tickers predeterminados con su descripción."""
    return {"tickers": TICKERS_DEFAULT}

@router_utils.get("/health", summary="Estado del servicio")
def health_check():
    return {"status": "ok", "message": "API de Análisis Financiero activa"}


# ─── MÓDULO 1: ANÁLISIS TÉCNICO ───────────────────────────────────────────────
router_tecnico = APIRouter(prefix="/api/tecnico", tags=["Módulo 1 · Análisis Técnico"])

@router_tecnico.post(
    "/indicadores",
    response_model=IndicadoresResponse,
    summary="Calcula indicadores técnicos (SMA, EMA, RSI, MACD, Bollinger, Estocástico)"
)
def get_indicadores(req: TecnicoRequest, svc: ServiceDep):
    """
    Descarga precios históricos y calcula los indicadores técnicos principales.
    Retorna los últimos 120 días de datos con todos los indicadores calculados.
    """
    return svc.get_tecnico(
        req.ticker, req.periodo.value,
        req.sma_corto, req.sma_largo,
        req.rsi_periodo, req.bb_periodo, req.bb_std
    )


# ─── MÓDULO 2: RENDIMIENTOS ───────────────────────────────────────────────────
router_rendimientos = APIRouter(prefix="/api/rendimientos", tags=["Módulo 2 · Rendimientos"])

@router_rendimientos.post(
    "/estadisticas",
    response_model=RendimientosResponse,
    summary="Estadísticas descriptivas y pruebas de normalidad de rendimientos"
)
def get_rendimientos(req: RendimientosRequest, svc: ServiceDep):
    """
    Calcula rendimientos simples o logarítmicos y entrega estadísticas completas:
    media, desviación, asimetría, curtosis, Jarque-Bera y Shapiro-Wilk.
    """
    return svc.get_rendimientos(req.ticker, req.periodo.value, req.tipo.value)


# ─── MÓDULO 3: GARCH ──────────────────────────────────────────────────────────
router_garch = APIRouter(prefix="/api/garch", tags=["Módulo 3 · ARCH/GARCH"])

@router_garch.post(
    "/volatilidad",
    response_model=GarchResponse,
    summary="Modela y pronostica volatilidad con ARCH(1), GARCH(1,1) y EGARCH(1,1)"
)
def get_garch(req: GarchRequest, svc: ServiceDep):
    """
    Ajusta tres especificaciones de modelos GARCH con la distribución seleccionada.
    Retorna comparativa AIC/BIC, pronóstico de volatilidad y diagnóstico de residuos.
    """
    return svc.get_garch(req.ticker, req.horizonte, req.distribucion.value)


# ─── MÓDULO 4: CAPM ───────────────────────────────────────────────────────────
router_capm = APIRouter(prefix="/api/capm", tags=["Módulo 4 · CAPM y Beta"])

@router_capm.post(
    "/beta",
    response_model=CapmResponse,
    summary="Estima Beta y retorno esperado bajo el modelo CAPM"
)
def get_capm(req: CapmRequest, svc: ServiceDep):
    """
    Obtiene la tasa libre de riesgo actual (^TNX), estima Beta por regresión OLS
    y calcula el retorno esperado según CAPM. Clasifica el activo como
    Agresivo (β>1.1), Neutro o Defensivo (β<0.9).
    """
    return svc.get_capm(req.ticker, req.benchmark, req.periodo.value)


# ─── MÓDULO 5: VaR / CVaR ─────────────────────────────────────────────────────
router_var = APIRouter(prefix="/api/var", tags=["Módulo 5 · VaR y CVaR"])

@router_var.post(
    "/calcular",
    response_model=VarResponse,
    summary="Calcula VaR (Paramétrico, Histórico, Montecarlo) y CVaR"
)
def get_var(req: VarRequest, svc: ServiceDep):
    """
    Cuantifica el riesgo de pérdida máxima bajo tres metodologías y calcula
    el Expected Shortfall (CVaR). Retorna valores porcentuales y en USD.
    """
    return svc.get_var(req.ticker, req.confianza, req.inversion, req.n_sims)


# ─── MÓDULO 6: MARKOWITZ ──────────────────────────────────────────────────────
router_markowitz = APIRouter(prefix="/api/markowitz", tags=["Módulo 6 · Markowitz"])

@router_markowitz.post(
    "/frontera",
    response_model=MarkowitzResponse,
    summary="Construye la frontera eficiente y portafolios óptimos (Markowitz)"
)
def get_markowitz(req: MarkowitzRequest, svc: ServiceDep):
    """
    Simula num_portafolios carteras aleatorias, construye la frontera eficiente
    e identifica el portafolio de Máximo Sharpe y Mínima Varianza.
    Incluye la matriz de correlación entre activos.
    """
    return svc.get_markowitz(req.tickers, req.num_portafolios, req.periodo.value)


# ─── MÓDULO 7: SEÑALES ────────────────────────────────────────────────────────
router_senales = APIRouter(prefix="/api/senales", tags=["Módulo 7 · Señales ★"])

@router_senales.post(
    "/panel",
    response_model=SenalesResponse,
    summary="Panel de señales algorítmicas (MACD, RSI, Bollinger, Medias Móviles)"
)
def get_senales(req: SenalesRequest, svc: ServiceDep):
    """
    Genera señales de COMPRA / VENTA / NEUTRAL basadas en cuatro indicadores técnicos.
    Incluye una señal global consolidada por mayoría de votos.
    """
    return svc.get_senales(req.ticker, req.rsi_up, req.rsi_down)


# ─── MÓDULO 8: MACRO ──────────────────────────────────────────────────────────
router_macro = APIRouter(prefix="/api/macro", tags=["Módulo 8 · Macro y Benchmark ★"])

@router_macro.post(
    "/benchmark",
    response_model=MacroResponse,
    summary="Comparativa del portafolio vs benchmark con métricas de desempeño"
)
def get_macro(req: MacroRequest, svc: ServiceDep):
    """
    Calcula rendimiento acumulado del portafolio igualmente ponderado vs S&P 500.
    Entrega Alpha, Tracking Error, Information Ratio y Máximo Drawdown.
    """
    return svc.get_macro(req.tickers, req.benchmark, req.periodo.value)