from sqlalchemy import Column, String, Float, DateTime, Integer, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from backend.database import Base


class Asset(Base):
    __tablename__ = "assets"

    ticker = Column(String, primary_key=True)
    nombre = Column(String)
    sector = Column(String)

    prices = relationship("Price", back_populates="asset")


class Price(Base):
    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, ForeignKey("assets.ticker"))
    fecha = Column(DateTime)
    close = Column(Float)

    asset = relationship("Asset", back_populates="prices")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String)
    tickers = Column(String)
    pesos = Column(String)
    creado = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String)
    fecha = Column(DateTime, default=datetime.utcnow)
    features = Column(String)
    prediccion = Column(Float)
    modelo = Column(String)