import yfinance as yf
from backend.ml.predictor import predecir


class MLService:

    def get_prediccion(self, ticker: str):

        df = yf.download(ticker, period="6mo", interval="1d", progress=False)

        df["ret_1d"] = df["Close"].pct_change()
        df["ret_5d"] = df["Close"].pct_change(5)
        df["vol_20d"] = df["ret_1d"].rolling(20).std()

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"] = 100 - 100/(1 + gain/loss)

        exp1 = df["Close"].ewm(span=12).mean()
        exp2 = df["Close"].ewm(span=26).mean()
        df["MACD"] = exp1 - exp2

        last = df.dropna().iloc[-1]

        features = {
            "ret_1d": float(last["ret_1d"]),
            "ret_5d": float(last["ret_5d"]),
            "vol_20d": float(last["vol_20d"]),
            "RSI": float(last["RSI"]),
            "MACD": float(last["MACD"]),
        }

        return predecir(ticker, features)