"""
dependencies.py
Inyección de dependencias con Depends() para servicios y configuración.
"""
from fastapi import Depends, HTTPException, status
from config import Settings, get_settings
from services import financial as fin_service

# ─── DEPENDENCIA: Configuración global ────────────────────────────────────────

def get_config(settings: Settings = Depends(get_settings)) -> Settings:
    """Provee la configuración de la aplicación a cualquier endpoint."""
    return settings


# ─── DEPENDENCIA: Servicio financiero ─────────────────────────────────────────

class FinancialService:
    """Clase de servicio inyectable que encapsula las funciones financieras."""

    def __init__(self, settings: Settings = Depends(get_settings)):
        self.settings = settings

    def get_tecnico(self, ticker, periodo, sma_corto, sma_largo,
                    rsi_periodo, bb_periodo, bb_std):
        try:
            return fin_service.calcular_tecnico(
                ticker, periodo, sma_corto, sma_largo, rsi_periodo, bb_periodo, bb_std
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    def get_rendimientos(self, ticker, periodo, tipo):
        try:
            return fin_service.calcular_rendimientos(ticker, periodo, tipo)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_garch(self, ticker, horizonte, distribucion):
        try:
            return fin_service.calcular_garch(ticker, horizonte, distribucion)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_capm(self, ticker, benchmark, periodo):
        try:
            return fin_service.calcular_capm(ticker, benchmark, periodo)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_var(self, ticker, confianza, inversion, n_sims):
        try:
            return fin_service.calcular_var(ticker, confianza, inversion, n_sims)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_markowitz(self, tickers, num_portafolios, periodo):
        try:
            return fin_service.calcular_markowitz(tickers, num_portafolios, periodo)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_senales(self, ticker, rsi_up, rsi_down):
        try:
            return fin_service.calcular_senales(ticker, rsi_up, rsi_down)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_macro(self, tickers, benchmark, periodo):
        try:
            return fin_service.calcular_macro(tickers, benchmark, periodo)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def get_financial_service(settings: Settings = Depends(get_settings)) -> FinancialService:
    return FinancialService(settings)