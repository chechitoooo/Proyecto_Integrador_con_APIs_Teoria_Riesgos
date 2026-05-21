import joblib
import pathlib
import numpy as np


class ModelPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            model_path = pathlib.Path(__file__).parent / "model.joblib"
            cls._model = joblib.load(model_path)
            print(f"[ML] Modelo cargado: {pathlib.Path(model_path).name}")
        return cls._instance

    def predict(self, features: np.ndarray):
        return self._model.predict(features)

    def predict_proba(self, features: np.ndarray):
        return self._model.predict_proba(features)


def predecir(ticker: str, features_data: dict) -> dict:
    predictor = ModelPredictor()
    X = np.array([[
        features_data["ret_1d"],
        features_data["ret_5d"],
        features_data["vol_20d"],
        features_data["RSI"],
        features_data["MACD"],
    ]])
    pred = int(predictor.predict(X)[0])
    proba = predictor.predict_proba(X)[0]
    return {
        "ticker": ticker,
        "prediccion": pred,
        "prob_sube": round(float(proba[1]), 4),
        "prob_baja": round(float(proba[0]), 4),
        "senal": "COMPRA" if pred == 1 else "VENTA",
    }
