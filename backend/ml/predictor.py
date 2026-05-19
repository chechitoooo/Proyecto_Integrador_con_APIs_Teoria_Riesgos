import joblib
import pathlib
import numpy as np

class ModeloSingleton:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            path = pathlib.Path(__file__).parent / "model.joblib"
            cls._instance = joblib.load(path)
        return cls._instance


def predecir(ticker: str, features: dict) -> dict:
    model = ModeloSingleton.get()

    X = np.array([[
        features["ret_1d"],
        features["ret_5d"],
        features["vol_20d"],
        features["RSI"],
        features["MACD"]
    ]])

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    return {
        "ticker": ticker,
        "prediccion": pred,
        "prob_sube": round(float(proba[1]), 4),
        "prob_baja": round(float(proba[0]), 4),
        "señal": "COMPRA" if pred == 1 else "VENTA",
    }