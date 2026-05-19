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
from backend.models import orm
from backend.config import get_settings

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    print("✅ Base de datos inicializada y lista.")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error interno: {str(exc)}"}
    )


# 🚨 IMPORTANTE: SIN PREFIX AQUÍ
app.include_router(router_utils)
app.include_router(router_tecnico)
app.include_router(router_rendimientos)
app.include_router(router_garch)
app.include_router(router_capm)
app.include_router(router_var)
app.include_router(router_markowitz)
app.include_router(router_senales)
app.include_router(router_macro)
app.include_router(router_renta_fija)
app.include_router(router_opciones)
app.include_router(router_stress)
app.include_router(router_ml)


@app.get("/", tags=["Root"])
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "status": "operativo"
    }