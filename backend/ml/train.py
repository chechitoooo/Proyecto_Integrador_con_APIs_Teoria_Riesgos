import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import yfinance as yf


TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA"
]


def build_features(df):

    df["ret_1d"] = df["Close"].pct_change()

    df["ret_5d"] = df["Close"].pct_change(5)

    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()

    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss

    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()

    ema26 = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26

    df["target"] = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)

    return df.dropna()


frames = []


for ticker in TICKERS:

    print(f"Descargando {ticker}...")

    try:

        df = yf.download(
            ticker,
            period="2y",
            progress=False,
            threads=False
        )

        if df.empty:
            print(f"? Sin datos para {ticker}")
            continue

        df = build_features(df)

        frames.append(df)

        print(f"? {ticker} OK")

    except Exception as e:

        print(f"? Error en {ticker}: {e}")


if len(frames) == 0:

    raise Exception("No se descargaron datos")


data = pd.concat(frames)


FEATURES = [
    "ret_1d",
    "ret_5d",
    "vol_20d",
    "RSI",
    "MACD"
]


X = data[FEATURES]

y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


print("Entrenando modelo...")


model.fit(X_train, y_train)


preds = model.predict(X_test)


print(classification_report(
    y_test,
    preds
))


joblib.dump(
    model,
    "backend/ml/model.joblib"
)


print("? Modelo guardado")
