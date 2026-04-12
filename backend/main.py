"""
main.py  —  Punto de entrada del backend FastAPI
Ejecutar: uvicorn main:app --reload --port 8000
Docs:     http://localhost:8000/docs
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from config import get_settings
from routers.endpoints import (
    router_utils, router_tecnico, router_rendimientos,
    router_garch, router_capm, router_var,
    router_markowitz, router_senales, router_macro,
)

settings = get_settings()

# ─── APLICACIÓN ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## Dashboard de Análisis Financiero — Backend API

API REST que expone los cálculos financieros para el dashboard de Teoría de Riesgo.

### Módulos disponibles
| # | Módulo | Endpoint |
|---|--------|---------|
| 1 | Análisis Técnico | `/api/tecnico/indicadores` |
| 2 | Rendimientos | `/api/rendimientos/estadisticas` |
| 3 | ARCH/GARCH | `/api/garch/volatilidad` |
| 4 | CAPM y Beta | `/api/capm/beta` |
| 5 | VaR y CVaR | `/api/var/calcular` |
| 6 | Markowitz | `/api/markowitz/frontera` |
| 7 | Señales ★ | `/api/senales/panel` |
| 8 | Macro & Benchmark ★ | `/api/macro/benchmark` |
    """,
    contact={"name": "Sergio D. Huertas / Sergio A. Prieto", "email": ""},
    license_info={"name": "MIT"},
)

# ─── CORS (permite que Streamlit consuma la API) ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # En producción: ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MIDDLEWARE: Tiempo de respuesta ──────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - start):.3f}s"
    return response

# ─── MANEJO GLOBAL DE ERRORES ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error interno: {str(exc)}"}
    )

# ─── REGISTRO DE ROUTERS ──────────────────────────────────────────────────────
for router in [
    router_utils, router_tecnico, router_rendimientos,
    router_garch, router_capm, router_var,
    router_markowitz, router_senales, router_macro,
]:
    app.include_router(router)

# ─── RAÍZ ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "endpoints": 9,
    }