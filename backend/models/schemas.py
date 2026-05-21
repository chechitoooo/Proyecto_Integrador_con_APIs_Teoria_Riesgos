from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Literal
from enum import Enum


class PeriodoEnum(str, Enum):
    one_year = "1y"
    two_years = "2y"
    five_years = "5y"
    max_hist = "max"


class TipoRendimientoEnum(str, Enum):
    simple = "Simple"
    logaritmico = "Logaritmico"


class DistribucionEnum(str, Enum):
    normal = "Normal"
    t_student = "t-Student"
    skewed_t = "Skewed t-Student"


# ─── REQUESTS ──────────────────────────────────────────────────────────────────

class TickerRequest(BaseModel):
    ticker: str
    periodo: PeriodoEnum = PeriodoEnum.two_years

    @field_validator("ticker")
    @classmethod
    def ticker_must_be_valid(cls, v: str) -> str:
        v = v.strip().upper()
        if not v or len(v) < 1:
            raise ValueError("El ticker no puede estar vacio")
        if len(v) > 10:
            raise ValueError("El ticker no puede tener mas de 10 caracteres")
        return v


class TecnicoRequest(TickerRequest):
    sma_corto: int = 20
    sma_largo: int = 50
    ema_periodo: int = 21
    rsi_periodo: int = 14
    bb_periodo: int = 20
    bb_std: float = 2.0
    stoch_k: int = 14


class RendimientosRequest(TickerRequest):
    tipo: TipoRendimientoEnum = TipoRendimientoEnum.logaritmico


class GarchRequest(BaseModel):
    ticker: str
    horizonte: int = 10
    distribucion: DistribucionEnum = DistribucionEnum.t_student
    lambda_ewma: float = 0.94


class CapmRequest(BaseModel):
    ticker: str
    benchmark: str = "^GSPC"
    periodo: PeriodoEnum = PeriodoEnum.two_years


class VarRequest(BaseModel):
    ticker: str
    confianza: float = 0.95
    inversion: float = 10000.0
    n_sims: int = 10000

    @field_validator("confianza")
    @classmethod
    def confianza_range(cls, v: float) -> float:
        if not (0.50 <= v <= 0.999):
            raise ValueError("La confianza debe estar entre 0.50 y 0.999")
        return v

    @field_validator("inversion")
    @classmethod
    def inversion_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("La inversion debe ser positiva")
        return v

    @field_validator("confianza")
    @classmethod
    def confianza_valida(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("confianza debe estar entre 0 y 1 (exclusivo)")
        return v


class MarkowitzRequest(BaseModel):
    tickers: List[str]
    num_portafolios: int = 10000
    periodo: PeriodoEnum = PeriodoEnum.two_years
    no_short_selling: bool = True

    @field_validator("tickers")
    @classmethod
    def at_least_two_tickers(cls, v: List[str]) -> List[str]:
        v = [t.strip().upper() for t in v]
        if len(v) < 2:
            raise ValueError("Se requieren al menos 2 activos para Markowitz")
        return v

    @field_validator("tickers")
    @classmethod
    def al_menos_dos_activos(cls, v):
        if len(v) < 2:
            raise ValueError("Se requieren al menos 2 activos")
        return [t.strip().upper() for t in v]


class SenalesRequest(BaseModel):
    ticker: str
    rsi_up: int = 70
    rsi_down: int = 30


class MacroRequest(BaseModel):
    tickers: List[str]
    benchmark: str = "^GSPC"
    periodo: PeriodoEnum = PeriodoEnum.one_year


class EwmaRequest(BaseModel):
    ticker: str
    lambda_: float = Field(0.94, alias="lambda")
    periodo: str = "2y"
    model_config = {"populate_by_name": True}


class BonoRequest(BaseModel):
    cupon_pct: float = Field(5.0, ge=0, le=20)
    vencimiento: int = Field(1, ge=1, le=30)
    valor_nominal: float = Field(1000.0, gt=0)
    frecuencia: int = Field(2, ge=1, le=4)


class OpcionRequest(BaseModel):
    ticker: str
    strike: float = Field(gt=0)
    vencimiento_dias: int = Field(1, ge=1, le=365)
    tasa_libre_riesgo: float = Field(0.045, ge=0, alias="tasa")
    model_config = {"populate_by_name": True}

    @field_validator("vencimiento_dias")
    @classmethod
    def vencimiento_valido(cls, v):
        if v < 1:
            raise ValueError("vencimiento_dias debe ser al menos 1")
        return v


class StressRequest(BaseModel):
    tickers: List[str]
    inversion: float = Field(100000, gt=0)
    confianza: float = Field(0.99, ge=0.9, le=0.999)


class PredictRequest(BaseModel):
    ticker: str


# ─── RESPONSES ─────────────────────────────────────────────────────────────────

class IndicadoresResponse(BaseModel):
    ticker: str
    periodo: str
    ultimo_precio: float
    retorno_periodo_pct: float
    rsi_actual: float
    volatilidad_diaria_pct: float
    sma_corto: Optional[float] = None
    sma_largo: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    datos: List[Dict]


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
    datos_rendimientos: List[Dict]


class GarchResponse(BaseModel):
    ticker: str
    distribucion: str
    comparativa_modelos: List[Dict]
    pronostico_volatilidad: List[float]
    volatilidad_condicional: List[float]
    jb_residuos_pvalor: float
    residuos_std: List[float]
    ewma_volatilidad: List[float]
    vol_muestral_rodante: List[float]
    fechas_vol: List[str]
    arch_lm_pvalor: float


class CapmResponse(BaseModel):
    ticker: str
    benchmark: str
    beta: float
    retorno_esperado_pct: float
    rf_anual_pct: float
    rm_anual_pct: float
    r_squared: float
    clasificacion: Literal["Agresivo", "Neutro", "Defensivo"]
    datos_regresion: List[Dict]
    alpha_jensen_pct: float
    retorno_acumulado_pct: float
    retorno_anual_pct: float
    volatilidad_anual_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    tracking_error_pct: float
    information_ratio: float
    varianza_total: float
    varianza_sistematica: float
    varianza_no_sistematica: float


class KupiecResult(BaseModel):
    excedencias: int
    excedencias_esp: float
    lr_uc: float
    p_valor: float
    aprueba: bool


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
    datos_rendimientos: List[float]
    kupiec_parametrico: KupiecResult
    kupiec_historico: KupiecResult
    kupiec_montecarlo: KupiecResult


class PortafolioOptimo(BaseModel):
    tipo: Literal["max_sharpe", "min_varianza"]
    retorno_anual_pct: float
    volatilidad_anual_pct: float
    sharpe_ratio: float
    pesos: Dict[str, float]


class MarkowitzResponse(BaseModel):
    tickers: List[str]
    matriz_correlacion: Dict[str, Dict[str, float]]
    frontera_eficiente: List[Dict]
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
    senales: List[Senal]
    senal_global: Literal["COMPRA", "VENTA", "NEUTRAL"]


class MacroResponse(BaseModel):
    rf_pct: float
    rendimiento_portafolio_pct: float
    rendimiento_benchmark_pct: float
    alpha_pct: float
    tracking_error_pct: float
    information_ratio: float
    max_drawdown_pct: float
    volatilidad_anual_pct: float
    portafolio_acumulado: List[Dict]
    benchmark_acumulado: List[Dict]


class EwmaResponse(BaseModel):
    ticker: str
    lambda_: float = Field(alias="lambda")
    volatilidad_ewma_anual_pct: float
    serie_vol: List[float]
    fechas: List[str]
    model_config = {"populate_by_name": True}


class CurvaResponse(BaseModel):
    beta0: float
    beta1: float
    beta2: float
    lambda_: float = Field(alias="lambda")
    rmse: Optional[float] = None
    n_puntos: Optional[int] = None
    curva: List[Dict]
    puntos: Optional[List[Dict]] = None
    model_config = {"populate_by_name": True}


class SensibilidadItem(BaseModel):
    shock_bp: int
    precio_real: float
    cambio_real_pct: float
    aprox_duracion_pct: float
    aprox_duracion_convexidad_pct: float


class BonoResponse(BaseModel):
    precio: float
    ytm_pct: float
    duracion_macaulay: float
    duracion_modificada: float
    convexidad: float
    cupon_pct: float
    vencimiento: int
    valor_nominal: float
    frecuencia: int
    sensibilidad: Optional[List[SensibilidadItem]] = None


class GreeksModel(BaseModel):
    delta_call: float
    delta_put: float
    gamma: float
    vega: float
    theta_call: float
    rho_call: float


class OpcionResponse(BaseModel):
    ticker: str
    precio_spot: float
    strike: float
    sigma: float
    T_anios: float
    call_price: float
    put_price: float
    greeks: GreeksModel
    paridad_put_call: Optional[Dict] = None
    volatilidad_implicita: Optional[float] = None
    curva_payoff: Optional[Dict] = None
    curvas_delta: Optional[List[Dict]] = None


class StressResponse(BaseModel):
    tickers: List[str]
    var_base_pct: float
    var_base_usd: float
    beta_portafolio: float
    escenarios: List[Dict]
    reverse_stress_shock: float
    heatmap_activos: Dict


class PredictResponse(BaseModel):
    ticker: str
    prediccion: int
    prob_sube: float
    prob_baja: float
    senal: str
