# ─────────────────────────────
# 5.1 TEST SUITE COMPLETA
# ─────────────────────────────

def test_health(client):
    r = client.get("/api/utils/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_tecnico(client):
    r = client.post("/api/tecnico/indicadores", json={
        "ticker": "AAPL",
        "periodo": "1y"
    })
    assert r.status_code == 200
    data = r.json()
    assert "rsi_actual" in data
    assert data["ticker"] == "AAPL"


def test_rendimientos(client):
    r = client.post("/api/rendimientos/estadisticas", json={
        "ticker": "MSFT",
        "periodo": "1y",
        "tipo": "Simple"
    })
    assert r.status_code == 200
    assert "media" in r.json()


def test_var(client):
    r = client.post("/api/var/calcular", json={
        "ticker": "MSFT",
        "confianza": 0.95,
        "inversion": 10000
    })
    assert r.status_code == 200
    data = r.json()

    assert "var_parametrico_diario_pct" in data
    assert "var_parametrico_anual_pct" in data
    assert "aprueba_kupiec" in data


def test_markowitz(client):
    r = client.post("/api/markowitz/frontera", json={
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "num_portafolios": 500
    })
    assert r.status_code == 200
    assert "portafolio_max_sharpe" in r.json()


def test_senales(client):
    r = client.post("/api/senales/panel", json={
        "ticker": "AAPL",
        "rsi_up": 70,
        "rsi_down": 30
    })
    assert r.status_code == 200
    assert "senales" in r.json()


def test_macro(client):
    r = client.post("/api/macro/benchmark", json={
        "tickers": ["AAPL", "MSFT"],
        "benchmark": "^GSPC",
        "periodo": "1y"
    })
    assert r.status_code == 200
    assert "alpha_pct" in r.json()


def test_ml_predict(client):
    r = client.post("/api/ml/predict", json={
        "ticker": "AAPL"
    })
    assert r.status_code == 200
    data = r.json()
    assert "prob_sube" in data
    assert "señal" in data