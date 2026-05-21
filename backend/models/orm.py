from sqlalchemy import Column, String, Float, DateTime, Integer, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database import Base


class Asset(Base):
    __tablename__ = "assets"
    ticker = Column(String, primary_key=True, index=True)
    nombre = Column(String, nullable=False)
    sector = Column(String, nullable=True)
    prices = relationship("Price", back_populates="asset", cascade="all, delete-orphan")


class Price(Base):
    __tablename__ = "prices"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, ForeignKey("assets.ticker"), nullable=False)
    fecha = Column(DateTime, nullable=False)
    close = Column(Float, nullable=False)
    asset = relationship("Asset", back_populates="prices")


class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String, nullable=False)
    tickers = Column(String, nullable=True)
    pesos = Column(String, nullable=True)
    creado = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    fecha = Column(DateTime, default=datetime.utcnow)
    features = Column(String, nullable=True)
    prediccion = Column(Float, nullable=True)
    modelo = Column(String, nullable=True)


class SignalLog(Base):
    __tablename__ = "signals_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    fecha = Column(DateTime, default=datetime.utcnow)
    indicador = Column(String, nullable=False)
    estado = Column(String, nullable=False)
    descripcion = Column(String, nullable=True)
    color = Column(String, nullable=True)
