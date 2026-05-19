"""
backend/main.py
Punto de entrada del backend FastAPI.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from backend.database import engine, Base
from backend.models import orm  # registra modelos en metadata
from backend.config import get_settings

# Routers
from backend.routers.endpoints import (
    router_utils,
    router_tecnico,
    router_rendimientos,
    router_garch,
    router_capm,
    router_var,
    router_markowitz,
    router_senales,
    router_macro,
    router_renta_fija,
    router_opciones,
    router_stress,
    router_ml
)

settings = get_settings()


# ───────────────────────────────
# LIFESPAN
# ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    print("✅ Base de datos inicializada y lista.")
    yield


# ───────────────────────────────
# APP
# ───────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)


# ───────────────────────────────
# CORS
# ───────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ───────────────────────────────
# MIDDLEWARE
# ───────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


# ───────────────────────────────
# GLOBAL ERROR HANDLER
# ───────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error interno: {str(exc)}"}
    )


# ───────────────────────────────
# ROUTERS (IMPORTANTE: PREFIX CORRECTO)
# ───────────────────────────────
app.include_router(router_utils)

app.include_router(router_tecnico, prefix="/api/tecnico")
app.include_router(router_rendimientos, prefix="/api/rendimientos")
app.include_router(router_garch, prefix="/api/garch")
app.include_router(router_capm, prefix="/api/capm")
app.include_router(router_var, prefix="/api/var")
app.include_router(router_markowitz, prefix="/api/markowitz")
app.include_router(router_senales, prefix="/api/senales")
app.include_router(router_macro, prefix="/api/macro")
app.include_router(router_renta_fija, prefix="/api/renta-fija")
app.include_router(router_opciones, prefix="/api/opciones")
app.include_router(router_stress, prefix="/api/stress")
app.include_router(router_ml, prefix="/api/ml")


# ───────────────────────────────
# ROOT
# ───────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "status": "operativo"
    }