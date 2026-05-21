"""
backend/routers/endpoints.py
Routers FastAPI para los 11 modulos + ML.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.models.schemas import (
    TecnicoRequest, RendimientosRequest, GarchRequest, CapmRequest,
    VarRequest, MarkowitzRequest, SenalesRequest, MacroRequest,
    EwmaRequest, BonoRequest, OpcionRequest, StressRequest,
    PredictRequest,
    IndicadoresResponse, RendimientosResponse, GarchResponse,
    CapmResponse, VarResponse, MarkowitzResponse, SenalesResponse, MacroResponse,
    EwmaResponse, CurvaResponse, BonoResponse, OpcionResponse, StressResponse,
    PredictResponse,
)

from backend.services.financial import (
    calcular_tecnico, calcular_rendimientos, calcular_garch,
    calcular_capm, calcular_var, calcular_markowitz,
    calcular_senales, calcular_macro,
    calcular_ewma, calcular_curva_rendimiento, calcular_bono,
    calcular_opciones, calcular_stress,
)

from backend.services.ml_service import MLService
from backend.models.orm import PredictionLog
from backend.database import get_db

router_utils = APIRouter(prefix="/api/utils", tags=["Utils"])
router_tecnico = APIRouter(prefix="/api/tecnico", tags=["Modulo 1 - Tecnico"])
router_rendimientos = APIRouter(prefix="/api/rendimientos", tags=["Modulo 2 - Rendimientos"])
router_garch = APIRouter(prefix="/api/garch", tags=["Modulo 3 - GARCH"])
router_capm = APIRouter(prefix="/api/capm", tags=["Modulo 4 - CAPM"])
router_var = APIRouter(prefix="/api/var", tags=["Modulo 5 - VaR"])
router_markowitz = APIRouter(prefix="/api/markowitz", tags=["Modulo 6 - Markowitz"])
router_senales = APIRouter(prefix="/api/senales", tags=["Modulo 7 - Senales"])
router_macro = APIRouter(prefix="/api/macro", tags=["Modulo 8 - Macro"])
router_renta_fija = APIRouter(prefix="/api/renta-fija", tags=["Modulo 9 - Renta Fija"])
router_opciones = APIRouter(prefix="/api/opciones", tags=["Modulo 10 - Opciones"])
router_stress = APIRouter(prefix="/api/stress", tags=["Modulo 11 - Stress Testing"])
router_ml = APIRouter(prefix="/api/ml", tags=["ML"])

svc = MLService()


# ─── UTILS ────────────────────────────────────────────────────────────────────

@router_utils.get("/health")
def health():
    return {"status": "ok", "message": "Backend operativo"}


@router_utils.get("/tickers")
def get_tickers():
    return {
        "tickers": {
            "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
            "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "NVIDIA",
            "JPM": "JPMorgan", "BAC": "Bank of America",
            "GLD": "Gold ETF", "BTC-USD": "Bitcoin",
        }
    }


# ─── MODULO 1 ─────────────────────────────────────────────────────────────────

@router_tecnico.post("/indicadores", response_model=IndicadoresResponse)
def indicadores(req: TecnicoRequest):
    try:
        return calcular_tecnico(req.ticker, req.periodo.value, req.sma_corto, req.sma_largo,
                                 req.rsi_periodo, req.bb_periodo, req.bb_std)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── MODULO 2 ─────────────────────────────────────────────────────────────────

@router_rendimientos.post("/estadisticas", response_model=RendimientosResponse)
def rendimientos(req: RendimientosRequest):
    try:
        return calcular_rendimientos(req.ticker, req.periodo.value, req.tipo.value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── MODULO 3 ─────────────────────────────────────────────────────────────────

@router_garch.post("/volatilidad", response_model=GarchResponse)
def volatilidad(req: GarchRequest):
    try:
        return calcular_garch(req.ticker, req.horizonte, req.distribucion.value, req.lambda_ewma)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router_garch.post("/ewma", response_model=EwmaResponse)
def ewma(req: EwmaRequest):
    try:
        return calcular_ewma(req.ticker, req.lambda_, req.periodo)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── MODULO 4 ─────────────────────────────────────────────────────────────────

@router_capm.post("/beta", response_model=CapmResponse)
def capm(req: CapmRequest):
    try:
        return calcular_capm(req.ticker, req.benchmark, req.periodo.value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── MODULO 5 ─────────────────────────────────────────────────────────────────

@router_var.post("/calcular", response_model=VarResponse)
def var_endpoint(req: VarRequest):
    try:
        return calcular_var(req.ticker, req.confianza, req.inversion, req.n_sims)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── MODULO 6 ─────────────────────────────────────────────────────────────────

@router_markowitz.post("/frontera", response_model=MarkowitzResponse)
def markowitz(req: MarkowitzRequest):
    try:
        return calcular_markowitz(req.tickers, req.num_portafolios, req.periodo.value, req.no_short_selling)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── MODULO 7 ─────────────────────────────────────────────────────────────────

@router_senales.post("/panel", response_model=SenalesResponse)
def senales(req: SenalesRequest):
    try:
        return calcular_senales(req.ticker, req.rsi_up, req.rsi_down)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── MODULO 8 ─────────────────────────────────────────────────────────────────

@router_macro.get("/indicadores")
def indicadores_macro():
    return {"rf_10y_pct": 0.0432, "cpi_pct": 0.032, "trm_cop_usd": 4100, "fed_funds_pct": 0.0525}


@router_macro.post("/benchmark", response_model=MacroResponse)
def macro(req: MacroRequest):
    try:
        return calcular_macro(req.tickers, req.benchmark, req.periodo.value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── MODULO 9 ─────────────────────────────────────────────────────────────────

@router_renta_fija.get("/curva", response_model=CurvaResponse)
def curva():
    try:
        return calcular_curva_rendimiento()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_renta_fija.post("/bono", response_model=BonoResponse)
def bono(req: BonoRequest):
    try:
        return calcular_bono(req.cupon_pct, req.vencimiento, req.valor_nominal, req.frecuencia)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── MODULO 10 ────────────────────────────────────────────────────────────────

@router_opciones.post("/valorar", response_model=OpcionResponse)
def opciones(req: OpcionRequest):
    try:
        return calcular_opciones(req.ticker, req.strike, req.vencimiento_dias, req.tasa_libre_riesgo)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── MODULO 11 ────────────────────────────────────────────────────────────────

@router_stress.post("/calcular", response_model=StressResponse)
def stress(req: StressRequest):
    try:
        return calcular_stress(req.tickers, req.inversion, req.confianza)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── ML ───────────────────────────────────────────────────────────────────────

@router_ml.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    result = svc.get_prediccion(req.ticker)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    log = PredictionLog(
        ticker=req.ticker,
        prediccion=result["prob_sube"],
        features=str(result),
        modelo="RandomForest",
    )
    db.add(log)
    db.commit()
    return result


@router_ml.get("/historial")
def historial(limit: int = 10, db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
    return logs
