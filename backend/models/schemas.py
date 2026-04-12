from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum


# ─── ENUMS ────────────────────────────────────────────────────────────────────

class PeriodoEnum(str, Enum):
    one_year  = "1y"
    two_years = "2y"
    five_years = "5y"
    max_hist  = "max"

class TipoRendimientoEnum(str, Enum):
    simple = "Simple"
    logaritmico = "Logarítmico"

class DistribucionEnum(str, Enum):
    normal   = "Normal"
    t_student = "t-Student"
    skewed_t  = "Skewed t-Student"


# ─── REQUESTS ─────────────────────────────────────────────────────────────────

class TickerRequest(BaseModel):
    ticker: str
    periodo: PeriodoEnum = PeriodoEnum.two_years

    @field_validator("ticker")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.strip().upper()


class TecnicoRequest(TickerRequest):
    sma_corto: int = 20
    sma_largo: int = 50
    rsi_periodo: int = 14
    bb_periodo: int = 20
    bb_std: float = 2.0


class RendimientosRequest(TickerRequest):
    tipo: TipoRendimientoEnum = TipoRendimientoEnum.logaritmico


class GarchRequest(BaseModel):
    ticker: str
    horizonte: int = 10
    distribucion: DistribucionEnum = DistribucionEnum.t_student


class CapmRequest(BaseModel):
    ticker: str
    benchmark: str = "^GSPC"
    periodo: PeriodoEnum = PeriodoEnum.two_years


class VarRequest(BaseModel):
    ticker: str
    confianza: float = 0.95
    inversion: float = 10000.0
    n_sims: int = 10000


class MarkowitzRequest(BaseModel):
    tickers: list[str]
    num_portafolios: int = 10000
    periodo: PeriodoEnum = PeriodoEnum.two_years


class SenalesRequest(BaseModel):
    ticker: str
    rsi_up: int = 70
    rsi_down: int = 30


class MacroRequest(BaseModel):
    tickers: list[str]
    benchmark: str = "^GSPC"
    periodo: PeriodoEnum = PeriodoEnum.one_year


# ─── RESPONSES ────────────────────────────────────────────────────────────────

class IndicadoresResponse(BaseModel):
    ticker: str
    periodo: str
    ultimo_precio: float
    retorno_periodo_pct: float
    rsi_actual: float
    volatilidad_diaria_pct: float
    sma_corto: Optional[float]
    sma_largo: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    datos: list[dict]


class RendimientosResponse(BaseModel):
    ticker: str
    tipo: str
    media: float
    desviacion: float
    asimetria: float
    curtosis: float
    jarque_bera_pvalor: float
    shapiro_pvalor: float
    es_normal_jb: bool
    es_normal_sw: bool
    datos_rendimientos: list[dict]


class GarchResponse(BaseModel):
    ticker: str
    distribucion: str
    comparativa_modelos: list[dict]
    pronostico_volatilidad: list[float]
    jb_residuos_pvalor: float
    residuos_std: list[float]


class CapmResponse(BaseModel):
    ticker: str
    benchmark: str
    beta: float
    retorno_esperado_pct: float
    rf_anual_pct: float
    rm_anual_pct: float
    r_squared: float
    clasificacion: Literal["Agresivo", "Neutro", "Defensivo"]
    datos_regresion: list[dict]


class VarResponse(BaseModel):
    ticker: str
    confianza: float
    inversion: float
    var_parametrico_diario_pct: float
    var_parametrico_anual_pct: float
    var_historico_diario_pct: float
    var_montecarlo_diario_pct: float
    cvar_diario_pct: float
    perdida_param_usd: float
    perdida_hist_usd: float
    perdida_mc_usd: float
    perdida_cvar_usd: float
    datos_rendimientos: list[float]


class PortafolioOptimo(BaseModel):
    tipo: Literal["max_sharpe", "min_varianza"]
    retorno_anual_pct: float
    volatilidad_anual_pct: float
    sharpe_ratio: float
    pesos: dict[str, float]


class MarkowitzResponse(BaseModel):
    tickers: list[str]
    matriz_correlacion: dict[str, dict[str, float]]
    frontera_eficiente: list[dict]
    portafolio_max_sharpe: PortafolioOptimo
    portafolio_min_varianza: PortafolioOptimo


class Senal(BaseModel):
    indicador: str
    estado: str
    descripcion: str
    color: Literal["green", "red", "blue"]


class SenalesResponse(BaseModel):
    ticker: str
    precio_actual: float
    rsi_actual: float
    senales: list[Senal]
    señal_global: Literal["COMPRA", "VENTA", "NEUTRAL"]


class MacroResponse(BaseModel):
    rf_pct: float
    rendimiento_portafolio_pct: float
    rendimiento_benchmark_pct: float
    alpha_pct: float
    tracking_error_pct: float
    information_ratio: float
    max_drawdown_pct: float
    volatilidad_anual_pct: float
    portafolio_acumulado: list[dict]
    benchmark_acumulado: list[dict]