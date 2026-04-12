from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "Dashboard Financiero API"
    app_version: str = "1.0.0"
    debug: bool = True
    default_period: str = "2y"
    default_ticker: str = "AAPL"
    rf_rate: float = 0.04
    monte_carlo_sims: int = 10000

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()