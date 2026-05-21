import yfinance as yf
import pandas as pd
import numpy as np
from backend.ml.predictor import predecir


class MLService:

    def get_prediccion(self, ticker: str):
        try:
            df = yf.download(ticker, period="6mo", auto_adjust=False, progress=False)
            if df is None or df.empty:
                raise ValueError(f"No data for ticker: {ticker}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)

            df["ret_1d"] = df["Close"].pct_change()
            df["ret_5d"] = df["Close"].pct_change(5)
            df["vol_20d"] = df["ret_1d"].rolling(20).std()
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df["RSI"] = 100 - 100 / (1 + gain / loss)
            exp1 = df["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2

            clean = df.dropna()
            if clean.empty:
                raise ValueError("Not enough data after feature engineering")
            last = clean.iloc[-1]
            features = {
                "ret_1d": float(last["ret_1d"]),
                "ret_5d": float(last["ret_5d"]),
                "vol_20d": float(last["vol_20d"]),
                "RSI": float(last["RSI"]),
                "MACD": float(last["MACD"]),
            }
            return predecir(ticker, features)
        except Exception as e:
            return {"error": str(e), "ticker": ticker}
