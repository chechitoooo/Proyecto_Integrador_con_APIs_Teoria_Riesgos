import pytest


class TestAPI:
    """Suite de tests para los endpoints del backend."""

    def test_health(self, client):
        r = client.get("/api/utils/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_tickers(self, client):
        r = client.get("/api/utils/tickers")
        assert r.status_code == 200
        assert "tickers" in r.json()

    def test_tecnico(self, client):
        r = client.post("/api/tecnico/indicadores", json={
            "ticker": "AAPL", "periodo": "1y",
            "sma_corto": 20, "sma_largo": 50
        })
        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert "rsi_actual" in data
        assert "ultimo_precio" in data

    def test_rendimientos(self, client):
        r = client.post("/api/rendimientos/estadisticas", json={
            "ticker": "AAPL", "periodo": "1y", "tipo": "Logaritmico"
        })
        assert r.status_code == 200
        data = r.json()
        assert "media" in data
        assert "desviacion" in data

    def test_garch(self, client):
        r = client.post("/api/garch/volatilidad", json={
            "ticker": "AAPL", "horizonte": 5, "distribucion": "t-Student"
        })
        assert r.status_code == 200
        data = r.json()
        assert "comparativa_modelos" in data
        assert len(data["comparativa_modelos"]) >= 3

    def test_capm(self, client):
        r = client.post("/api/capm/beta", json={
            "ticker": "AAPL", "benchmark": "^GSPC", "periodo": "1y"
        })
        assert r.status_code == 200
        data = r.json()
        assert "beta" in data
        assert "alpha_jensen_pct" in data

    def test_var(self, client):
        r = client.post("/api/var/calcular", json={
            "ticker": "AAPL", "confianza": 0.95, "inversion": 10000
        })
        assert r.status_code == 200
        data = r.json()
        assert "var_parametrico_diario_pct" in data
        assert "var_historico_diario_pct" in data
        assert "var_montecarlo_diario_pct" in data
        assert "cvar_diario_pct" in data
        assert "kupiec_parametrico" in data
        assert "kupiec_historico" in data
        assert "kupiec_montecarlo" in data

    def test_markowitz(self, client):
        r = client.post("/api/markowitz/frontera", json={
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "num_portafolios": 500, "periodo": "1y"
        })
        assert r.status_code == 200
        data = r.json()
        assert "portafolio_max_sharpe" in data
        assert "portafolio_min_varianza" in data
        assert "matriz_correlacion" in data

    def test_senales(self, client):
        r = client.post("/api/senales/panel", json={
            "ticker": "AAPL", "rsi_up": 70, "rsi_down": 30
        })
        assert r.status_code == 200
        data = r.json()
        assert "senales" in data
        assert len(data["senales"]) >= 4
        assert "senal_global" in data   # CORREGIDO: era "señal_global"

    def test_macro(self, client):
        r = client.get("/api/macro/indicadores")
        assert r.status_code == 200
        assert "rf_10y_pct" in r.json()

    def test_benchmark(self, client):
        r = client.post("/api/macro/benchmark", json={
            "tickers": ["AAPL", "MSFT"], "benchmark": "^GSPC", "periodo": "1y"
        })
        assert r.status_code == 200
        assert "alpha_pct" in r.json()

    def test_curva(self, client):
        r = client.get("/api/renta-fija/curva")
        assert r.status_code == 200
        data = r.json()
        assert "curva" in data

    def test_bono(self, client):
        r = client.post("/api/renta-fija/bono", json={
            "cupon_pct": 5, "vencimiento": 10,
            "valor_nominal": 1000, "frecuencia": 2
        })
        assert r.status_code == 200
        data = r.json()
        assert "precio" in data
        assert "duracion_modificada" in data

    def test_opciones(self, client):
        r = client.post("/api/opciones/valorar", json={
            "ticker": "AAPL", "strike": 150,
            "vencimiento_dias": 30, "tasa_libre_riesgo": 0.045
        })
        assert r.status_code == 200
        data = r.json()
        assert "call_price" in data
        assert "put_price" in data
        assert "greeks" in data

    def test_stress(self, client):
        r = client.post("/api/stress/calcular", json={
            "tickers": ["AAPL", "MSFT"], "inversion": 100000, "confianza": 0.99
        })
        assert r.status_code == 200
        data = r.json()
        assert "escenarios" in data
        assert len(data["escenarios"]) >= 3

    def test_ml_predict(self, client):
        r = client.post("/api/ml/predict", json={"ticker": "AAPL"})
        assert r.status_code == 200
        data = r.json()
        assert "prob_sube" in data
        assert "prob_baja" in data
        assert "senal" in data   # CORREGIDO: era "señal"

    def test_var_validator_confianza(self, client):
        """Prueba que @field_validator rechace confianza invalida (HTTP 422)."""
        r = client.post("/api/var/calcular", json={
            "ticker": "AAPL", "confianza": 0.3, "inversion": 10000
        })
        assert r.status_code == 422

    def test_markowitz_validator_tickers(self, client):
        """Prueba que @field_validator rechace menos de 2 tickers (HTTP 422)."""
        r = client.post("/api/markowitz/frontera", json={
            "tickers": ["AAPL"], "num_portafolios": 500
        })
        assert r.status_code == 422
