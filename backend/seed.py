from backend.database import engine, SessionLocal, Base
from backend.models.orm import Asset, Price
import yfinance as yf

TICKERS = {
    "AAPL": ("Apple Inc.", "Tecnología"),
    "MSFT": ("Microsoft", "Software"),
    "GOOGL": ("Alphabet", "Publicidad/IA"),
    "AMZN": ("Amazon", "E-commerce"),
    "TSLA": ("Tesla", "Vehículos Eléctricos"),
    "NVDA": ("NVIDIA", "Semiconductores"),
    "JPM": ("JPMorgan", "Financiero"),
    "BAC": ("Bank of America", "Financiero"),
    "GLD": ("SPDR Gold ETF", "Commodities"),
    "BTC-USD": ("Bitcoin", "Criptomonedas"),
}

def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    for ticker, (nombre, sector) in TICKERS.items():

        if not db.query(Asset).filter_by(ticker=ticker).first():
            db.add(Asset(ticker=ticker, nombre=nombre, sector=sector))

        df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)

        for fecha, row in df.iterrows():
            close_value = row["Close"]

            if hasattr(close_value, "iloc"):
                close_value = close_value.iloc[0]

            db.merge(
                Price(
                    ticker=ticker,
                    fecha=fecha,
                    close=float(close_value)
                )
            )

    db.commit()
    db.close()

    print("✅ Seed completo")


if __name__ == "__main__":
    seed()