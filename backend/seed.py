"""
backend/seed.py
Carga inicial de activos y precios históricos en la BD.
Ejecutar: python -m backend.seed
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import engine, SessionLocal, Base
from backend.models.orm import Asset, Price
import yfinance as yf
import pandas as pd

TICKERS = {
    "AAPL": ("Apple Inc.", "Tecnología"), "MSFT": ("Microsoft", "Software"),
    "GOOGL": ("Alphabet", "Publicidad/IA"), "AMZN": ("Amazon", "E-commerce"),
    "TSLA": ("Tesla", "Vehículos Eléctricos"), "NVDA": ("NVIDIA", "Semiconductores"),
    "JPM": ("JPMorgan", "Financiero"), "BAC": ("Bank of America", "Financiero"),
    "GLD": ("SPDR Gold ETF", "Commodities"), "BTC-USD": ("Bitcoin", "Criptomonedas")
}

def seed():
    print("🔧 Inicializando base de datos...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for ticker, (nombre, sector) in TICKERS.items():
            print(f"📥 Procesando {ticker}...")
            if not db.query(Asset).filter_by(ticker=ticker).first():
                db.add(Asset(ticker=ticker, nombre=nombre, sector=sector))
                db.flush()
            try:
                df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
                if df.empty: print(f"  ⚠️ No se encontraron datos para {ticker}"); continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                count = 0
                for fecha, row in df.iterrows():
                    fecha_sin_tz = fecha.tz_localize(None) if fecha.tzinfo else fecha
                    close_val = float(row["Close"])
                    if not db.query(Price).filter_by(ticker=ticker, fecha=fecha_sin_tz).first():
                        db.add(Price(ticker=ticker, fecha=fecha_sin_tz, close=close_val))
                        count += 1
                db.commit()
                print(f"  ✅ {count} registros nuevos insertados.")
            except Exception as e:
                print(f"  ❌ Error descargando {ticker}: {e}")
                db.rollback()
        print("✅ Seed completado exitosamente.")
    except Exception as e:
        print(f"❌ Error crítico en seed: {e}")
        db.rollback()
    finally: db.close()

if __name__ == "__main__": seed()