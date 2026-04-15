# 📊 Dashboard de Análisis Financiero — Teoría de Riesgo

**Proyecto Final · Teoría de Riesgo · Profesor: Javier Sierra**

**Autores:** Sergio David Huertas Ramírez · Sergio Andrés Prieto Orjuela

---

## Descripción

Dashboard interactivo de análisis financiero con arquitectura desacoplada:

- **Backend (FastAPI):** 9 endpoints REST con toda la lógica de cálculo financiero, validación Pydantic, inyección de dependencias y documentación automática en `/docs`.
- **Frontend (Streamlit):** Dashboard que consume el backend exclusivamente vía HTTP — no ejecuta cálculos directamente.
- **Datos:** Yahoo Finance en tiempo real a través de `yfinance`. **No se requieren API keys.**

| # | Módulo | Descripción |
|---|--------|-------------|
| 1 | Análisis Técnico | SMA, EMA, RSI, MACD, Bollinger |
| 2 | Rendimientos | Estadísticas descriptivas, Jarque-Bera, Shapiro-Wilk, Q-Q |
| 3 | ARCH/GARCH | ARCH(1), GARCH(1,1), EGARCH(1,1), pronóstico de volatilidad |
| 4 | CAPM y Beta | Regresión OLS, Beta, retorno esperado, clasificación |
| 5 | VaR y CVaR | Paramétrico, Histórico, Montecarlo (10k sims), Expected Shortfall |
| 6 | Markowitz | Frontera eficiente, Máximo Sharpe, Mínima Varianza |
| 7 ★ | Señales | Panel semáforo: MACD, RSI, Bollinger, Medias Móviles |
| 8 ★ | Macro y Benchmark | Alpha, Tracking Error, Information Ratio vs S&P 500 |

---

## Estructura del repositorio

```
├── backend/
│   ├── main.py              # Punto de entrada FastAPI
│   ├── config.py            # BaseSettings + .env
│   ├── dependencies.py      # Inyección de dependencias (Depends)
│   ├── .env                 # Variables de entorno
│   ├── requirements.txt
│   ├── models/schemas.py    # Modelos Pydantic (Request + Response)
│   ├── routers/endpoints.py # 9 endpoints por módulo
│   └── services/financial.py# Lógica de cálculo financiero
├── frontend/
│   ├── app.py               # Dashboard Streamlit
│   
├── requirements.txt         # Dependencias globales (referencia)
├── .gitignore
└── README.md
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/chechitoooo/proyecto-riesgo.git
cd proyecto-riesgo

# 2. Entorno virtual y dependencias — Backend
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Entorno virtual y dependencias — Frontend (nueva terminal)
cd frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Variables de entorno

No se requieren API keys externas. Solo ajusta `backend/.env`:

```env
APP_NAME="Dashboard Financiero API"
APP_VERSION="1.0.0"
DEBUG=True
DEFAULT_PERIOD="2y"
DEFAULT_TICKER="AAPL"
RF_RATE=0.04
MONTE_CARLO_SIMS=10000
```

Si el archivo no existe, la app usa los valores por defecto de `config.py`.

---

## Ejecución

**Terminal 1 — Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
source venv/bin/activate
streamlit run app.py
```

- Dashboard: [http://localhost:8501](http://localhost:8501)
- API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

> El backend debe estar corriendo **antes** de abrir el frontend.

---

## Endpoints principales

| Método | Endpoint | Módulo |
|--------|----------|--------|
| `GET`  | `/api/utils/health` | Health check |
| `GET`  | `/api/utils/tickers` | Activos disponibles |
| `POST` | `/api/tecnico/indicadores` | Módulo 1 — Técnico |
| `POST` | `/api/rendimientos/estadisticas` | Módulo 2 — Rendimientos |
| `POST` | `/api/garch/volatilidad` | Módulo 3 — GARCH |
| `POST` | `/api/capm/beta` | Módulo 4 — CAPM |
| `POST` | `/api/var/calcular` | Módulo 5 — VaR / CVaR |
| `POST` | `/api/markowitz/frontera` | Módulo 6 — Markowitz |
| `POST` | `/api/senales/panel` | Módulo 7 — Señales |
| `POST` | `/api/macro/benchmark` | Módulo 8 — Macro |

Documentación completa con schemas y ejemplos: **[localhost:8000/docs](http://localhost:8000/docs)**

---

## Activos seleccionados

| Ticker | Empresa | Sector | Justificación |
|--------|---------|--------|---------------|
| `AAPL` | Apple | Tecnología | Mayor cap. del S&P 500; referente de análisis técnico |
| `MSFT` | Microsoft | Software / Cloud | Líder en nube; ideal para CAPM y correlaciones |
| `GOOGL` | Alphabet | Publicidad / IA | Contraste sectorial en Markowitz vs AAPL/MSFT |
| `AMZN` | Amazon | E-commerce / Cloud | Diversificación real: retail + infraestructura |
| `TSLA` | Tesla | Vehículos Eléctricos | Beta elevado; demuestra activos agresivos y alta volatilidad |
| `NVDA` | NVIDIA | Semiconductores / IA | Boom de IA; mayor crecimiento reciente, alto VaR |
| `JPM` | JPMorgan | Sector Financiero | Comportamiento anticíclico respecto a tech |
| `BAC` | Bank of America | Sector Financiero | Sensibilidad a tasas; complementa JPM |
| `GLD` | SPDR Gold ETF | Materias Primas | Activo refugio; reduce volatilidad en Markowitz |
| `BTC-USD` | Bitcoin | Criptomonedas | Alta asimetría y colas pesadas; ilustra hechos estilizados |

**Criterios:** Diversificación sectorial, contraste de perfiles de riesgo (β bajo a alto), liquidez y más de 5 años de historia en Yahoo Finance.

---

## Uso de herramientas de IA

Durante el desarrollo se utilizó **Claude (Anthropic)** como asistente para la generación del esqueleto de la arquitectura FastAPI, corrección de bugs en Streamlit (`DeltaGenerator`), diseño del CSS.

Todo el código generado fue revisado, comprendido y ajustado por el equipo.
