"""
backend/routers/endpoints.py
Routers FastAPI para los 11 módulos del proyecto + ML 4.3.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

# ─── EXISTENTE ───────────────────────────────────────────────────────────────
from backend.models.schemas import (
    TecnicoRequest, RendimientosRequest, GarchRequest, CapmRequest,
    VarRequest, MarkowitzRequest, SenalesRequest, MacroRequest,
    EwmaRequest, BonoRequest, OpcionRequest, StressRequest,
    PredictRequest,
    IndicadoresResponse, RendimientosResponse, GarchResponse,
    CapmResponse, VarResponse, MarkowitzResponse, SenalesResponse, MacroResponse,
    EwmaResponse, CurvaResponse, BonoResponse, OpcionResponse, StressResponse,
    PredictResponse
)

from backend.services.financial import (
    calcular_tecnico, calcular_rendimientos, calcular_garch,
    calcular_capm, calcular_var, calcular_markowitz,
    calcular_senales, calcular_macro,
    calcular_ewma, calcular_curva_rendimiento, calcular_bono,
    calcular_opciones, calcular_stress
)

# ─── ML 4.3 ──────────────────────────────────────────────────────────────────
from backend.services.ml_service import MLService
from backend.models.prediction_log import PredictionLog
from backend.db.session import get_db

# ─── ROUTERS EXISTENTES ──────────────────────────────────────────────────────
router_utils = APIRouter(prefix="/api/utils", tags=["Utils"])
router_tecnico = APIRouter(prefix="/api/tecnico", tags=["Módulo 1 · Técnico"])
router_rendimientos = APIRouter(prefix="/api/rendimientos", tags=["Módulo 2 · Rendimientos"])
router_garch = APIRouter(prefix="/api/garch", tags=["Módulo 3 · GARCH"])
router_capm = APIRouter(prefix="/api/capm", tags=["Módulo 4 · CAPM"])
router_var = APIRouter(prefix="/api/var", tags=["Módulo 5 · VaR"])
router_markowitz = APIRouter(prefix="/api/markowitz", tags=["Módulo 6 · Markowitz"])
router_senales = APIRouter(prefix="/api/senales", tags=["Módulo 7 · Señales"])
router_macro = APIRouter(prefix="/api/macro", tags=["Módulo 8 · Macro"])

router_renta_fija = APIRouter(prefix="/api/renta-fija", tags=["Módulo 9 · Renta Fija"])
router_opciones = APIRouter(prefix="/api/opciones", tags=["Módulo 10 · Opciones"])
router_stress = APIRouter(prefix="/api/stress", tags=["Módulo 11 · Stress Testing"])

# ─── ML ROUTER (NUEVO 4.3) ───────────────────────────────────────────────────
router_ml = APIRouter(prefix="/api/ml", tags=["ML 4.3"])

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
            "GLD": "Gold ETF", "BTC-USD": "Bitcoin"
        }
    }

# ─── ML 4.3 ENDPOINT ─────────────────────────────────────────────────────────
@router_ml.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):

    result = svc.get_prediccion(req.ticker)

    log = PredictionLog(
        ticker=req.ticker,
        prediccion=result["prob_sube"],
        features=str(result),
        modelo="RandomForest"
    )

    db.add(log)
    db.commit()

    return result


@router_ml.get("/historial")
def historial(limit: int = 10, db: Session = Depends(get_db)):

    logs = db.query(PredictionLog)\
        .order_by(PredictionLog.id.desc())\
        .limit(limit)\
        .all()

    return logs


# ─── MÓDULO 1: TÉCNICO ───────────────────────────────────────────────────────
@router_tecnico.post("/indicadores", response_model=IndicadoresResponse)
def indicadores(req: TecnicoRequest):
    try:
        return calcular_tecnico(
            req.ticker, req.periodo.value,
            req.sma_corto, req.sma_largo,
            req.rsi_periodo, req.bb_periodo, req.bb_std
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ─── MÓDULO 2: RENDIMIENTOS ──────────────────────────────────────────────────
@router_rendimientos.post("/estadisticas", response_model=RendimientosResponse)
def rendimientos(req: RendimientosRequest):
    try:
        return calcular_rendimientos(req.ticker, req.periodo.value, req.tipo.value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ─── MÓDULO 3: GARCH ─────────────────────────────────────────────────────────
@router_garch.post("/volatilidad", response_model=GarchResponse)
def volatilidad(req: GarchRequest):
    try:
        return calcular_garch(req.ticker, req.horizonte, req.distribucion.value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router_garch.post("/ewma", response_model=EwmaResponse)
def ewma(req: EwmaRequest):
    try:
        return calcular_ewma(req.ticker, req.lambda_, req.periodo)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ─── MÓDULO 4: CAPM ──────────────────────────────────────────────────────────
@router_capm.post("/beta", response_model=CapmResponse)
def capm(req: CapmRequest):
    try:
        return calcular_capm(req.ticker, req.benchmark, req.periodo.value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ─── MÓDULO 5: VaR ───────────────────────────────────────────────────────────
@router_var.post("/calcular", response_model=VarResponse)
def var(req: VarRequest):
    try:
        return calcular_var(req.ticker, req.confianza, req.inversion, req.n_sims)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ─── MÓDULO 6: MARKOWITZ ─────────────────────────────────────────────────────
@router_markowitz.post("/frontera", response_model=MarkowitzResponse)
def markowitz(req: MarkowitzRequest):
    try:
        return calcular_markowitz(req.tickers, req.num_portafolios, req.periodo.value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ─── MÓDULO 7: SEÑALES ───────────────────────────────────────────────────────
@router_senales.post("/panel", response_model=SenalesResponse)
def senales(req: SenalesRequest):
    try:
        return calcular_senales(req.ticker, req.rsi_up, req.rsi_down)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ─── MÓDULO 8: MACRO ─────────────────────────────────────────────────────────
@router_macro.post("/benchmark", response_model=MacroResponse)
def macro(req: MacroRequest):
    try:
        return calcular_macro(req.tickers, req.benchmark, req.periodo.value)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ─── MÓDULO 9: RENTA FIJA ────────────────────────────────────────────────────
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

# ─── MÓDULO 10: OPCIONES ─────────────────────────────────────────────────────
@router_opciones.post("/valorar", response_model=OpcionResponse)
def opciones(req: OpcionRequest):
    try:
        return calcular_opciones(req.ticker, req.strike, req.vencimiento_dias, req.tasa_libre_riesgo)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ─── MÓDULO 11: STRESS ───────────────────────────────────────────────────────
@router_stress.post("/calcular", response_model=StressResponse)
def stress(req: StressRequest):
    try:
        return calcular_stress(req.tickers, req.inversion, req.confianza)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))