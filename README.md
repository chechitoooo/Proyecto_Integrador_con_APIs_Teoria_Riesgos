# RiskLab USTA — Proyecto Integrador Teoría del Riesgo

**Autores:** Sergio David Huertas Ramírez · Sergio Andrés Prieto Orjuela  
**Profesor:** Javier Mauricio Sierra  
**Materia:** Teoría del Riesgo · Python para APIs e IA  
**Universidad Santo Tomás** — Facultad de Estadística — 2026

---

## Descripción

Sistema integral de análisis de riesgo financiero compuesto por un backend FastAPI con persistencia en SQLite, un componente de machine learning (Random Forest + patrón Singleton) y un tablero interactivo Streamlit. El sistema permite analizar un portafolio de 10 activos aplicando 11 módulos de riesgo: indicadores técnicos, rendimientos, volatilidad (EWMA + GARCH), CAPM, VaR con backtesting Kupiec, optimización de Markowitz con programación cuadrática, señales algorítmicas, comparativa macro/benchmark, valoración de renta fija y opciones, y stress testing.

## Arquitectura — Cinco Capas

| Capa | Componente | Tecnología |
|------|-----------|------------|
| **1. Datos y Persistencia** | Ingesta desde Yahoo Finance + FRED, cache en SQLite, ORM | Python, yfinance, SQLAlchemy |
| **2. Análisis de Riesgo** | Indicadores técnicos, rendimientos, volatilidad, CAPM, VaR, Markowitz | pandas, numpy, scipy, arch, cvxpy, statsmodels |
| **3. Renta Fija y Derivados** | Curva Nelson-Siegel, duración/convexidad, Black-Scholes, Greeks, Stress | scipy.optimize, fórmulas analíticas |
| **4. Machine Learning** | Random Forest classifier + patrón Singleton + log de predicciones | scikit-learn, joblib |
| **5. Frontend** | Tablero interactivo con 11 módulos, gráficos Plotly | Streamlit, Plotly |

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/chechitoooo/Proyecto_Integrador_con_APIs_Teoria_Riesgos.git
cd Proyecto_Integrador_con_APIs_Teoria_Riesgos

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Variables de Entorno

No se requieren API keys. El proyecto utiliza **yfinance** (acceso gratuito a Yahoo Finance sin autenticación). La tasa libre de riesgo se obtiene del treasury yield ^TNX vía yfinance. Para configurar la URL del backend desde el frontend, modificar la variable `API_BASE` en `frontend/app.py` (por defecto: `http://localhost:8000`).

## Ejecución

### Backend (FastAPI)

```bash
uvicorn backend.main:app --reload --port 8000
```

La documentación interactiva estará disponible en:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

Abrir http://localhost:8501 en el navegador.

> Si el backend corre en otro puerto, configurar con variable de entorno:
> ```bash
> set API_BASE=http://localhost:8001 && streamlit run frontend/app.py
> ```

## Modelo Machine Learning

### Propósito Analítico

Clasificación de la dirección del movimiento diario del precio (sube/baja) utilizando features técnicas derivadas de los datos del proyecto. El modelo actúa como filtro complementario al análisis técnico tradicional.

### Features
- `ret_1d`: retorno diario
- `ret_5d`: retorno a 5 días
- `vol_20d`: volatilidad rodante de 20 días
- `RSI`: Relative Strength Index (14 períodos)
- `MACD`: Moving Average Convergence Divergence

### Algoritmo
Random Forest Classifier con 200 estimadores (scikit-learn).

### Entrenamiento

```bash
python backend/ml/train.py
```

El modelo se entrena con datos históricos de 5 activos (AAPL, MSFT, GOOGL, AMZN, TSLA) a 2 años y se serializa en `backend/ml/model.joblib`. El modelo incluido en el repositorio ya está entrenado y listo para usar.

### Servicio

El modelo se carga usando el **patrón Singleton** (carga única al iniciar el servidor) y se expone vía el endpoint `POST /api/ml/predict`. Cada predicción se persiste en la tabla `prediction_logs` para trazabilidad. Un decorador propio (`@log_latency`) mide el tiempo de inferencia en cada request.

### Desempeño

| Métrica | Valor |
|---------|-------|
| Accuracy | 52% |
| Precisión (sube) | 54% |
| Recall (sube) | 60% |
| F1-Score | 57% |

## Activos Seleccionados

| Ticker | Nombre | Sector |
|--------|--------|--------|
| AAPL | Apple Inc. | Tecnología |
| MSFT | Microsoft Corp. | Tecnología |
| GOOGL | Alphabet Inc. | Tecnología |
| AMZN | Amazon.com Inc. | Tecnología/Consumo |
| TSLA | Tesla Inc. | Tecnología/Automotriz |
| NVDA | NVIDIA Corp. | Tecnología/Semiconductores |
| JPM | JPMorgan Chase | Financiero |
| BAC | Bank of America | Financiero |
| GLD | SPDR Gold ETF | Commodities |
| BTC-USD | Bitcoin USD | Criptoactivo |

**Justificación:** Se seleccionaron activos de distintos sectores (tecnología, financiero, commodities, criptoactivos) para evaluar el impacto de la diversificación sectorial en el perfil de riesgo del portafolio y analizar cómo diferentes betas, volatilidades y correlaciones interactúan en la frontera eficiente de Markowitz.

## Endpoints de la API

| Endpoint | Método | Módulo | Descripción |
|----------|--------|--------|-------------|
| `/api/utils/health` | GET | — | Estado del backend |
| `/api/utils/tickers` | GET | — | Lista de activos disponibles |
| `/api/utils/cache-status` | GET | — | Estado del cache de datos |
| `/api/tecnico/indicadores` | POST | 1 | Indicadores técnicos (6) |
| `/api/rendimientos/estadisticas` | POST | 2 | Rendimientos y pruebas normalidad |
| `/api/garch/volatilidad` | POST | 3 | GARCH + EWMA + pronóstico |
| `/api/garch/ewma` | POST | 3 | EWMA individual |
| `/api/capm/beta` | POST | 4 | Beta, CAPM, desempeño completo |
| `/api/var/calcular` | POST | 5 | VaR (3 métodos) + CVaR + Kupiec |
| `/api/markowitz/frontera` | POST | 6 | Frontera eficiente (QP) |
| `/api/senales/panel` | POST | 7 | Señales algorítmicas |
| `/api/senales/historial` | GET | 7 | Historial de señales persistido |
| `/api/macro/indicadores` | GET | 8 | Indicadores macroeconómicos |
| `/api/macro/benchmark` | POST | 8 | Comparativa vs benchmark |
| `/api/renta-fija/curva` | GET | 9 | Curva Nelson-Siegel |
| `/api/renta-fija/bono` | POST | 9 | Valoración bono + sensibilidad |
| `/api/opciones/valorar` | POST | 10 | Black-Scholes + Greeks anidados |
| `/api/stress/calcular` | POST | 11 | Stress testing |
| `/api/ml/predict` | POST | ML | Predicción ML |
| `/api/ml/historial` | GET | ML | Historial de predicciones |

## Estructura del Proyecto

```
proyecto/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── config.py            # BaseSettings
│   ├── database.py          # SQLAlchemy engine + session
│   ├── db/
│   │   └── __init__.py
│   ├── models/
│   │   ├── orm.py           # Asset, Price, Portfolio, PredictionLog, SignalLog
│   │   └── schemas.py       # Pydantic v2 (requests + responses)
│   ├── services/
│   │   ├── financial.py     # 11 módulos de lógica financiera
│   │   └── ml_service.py    # Servicio ML
│   ├── ml/
│   │   ├── train.py         # Entrenamiento offline
│   │   ├── predictor.py     # Singleton predictor + @log_latency
│   │   └── model.joblib     # Modelo serializado
│   └── routers/
│       ├── endpoints.py     # 16 endpoints REST
│       └── dependencies.py  # Depends
├── frontend/
│   └── app.py               # Dashboard Streamlit (11 módulos)
├── requirements.txt
└── Informe_Ejecutivo.html   # Informe ejecutivo del proyecto
```

## Stack Tecnológico

- **Python** 3.11+
- **Backend:** FastAPI, Pydantic v2, SQLAlchemy, uvicorn
- **Análisis:** pandas, numpy, scipy, statsmodels, arch, cvxpy
- **ML:** scikit-learn, joblib
- **Frontend:** Streamlit, Plotly
- **Datos:** yfinance, requests

## Uso de Herramientas de IA

Durante el desarrollo de este proyecto se utilizó **OpenCode** (DeepSeek v4) como asistente de programación para:

- Depuración y corrección de errores en tiempo de ejecución
- Refactorización del código backend (servicios financieros, schemas, ORM)
- Rediseño completo del frontend (CSS, tipografía, paleta de colores)
- Implementación de funcionalidades avanzadas: Kupiec en 3 métodos, Greeks anidados, volatilidad implícita por Newton-Raphson, sensibilidad de bonos con 3 aproximaciones, descomposición de varianza CAPM
- Verificación integral contra la rúbrica de evaluación

Todo el código fue revisado, comprendido y validado por los autores. La IA actuó como acelerador del desarrollo, no como sustituto del criterio técnico.
