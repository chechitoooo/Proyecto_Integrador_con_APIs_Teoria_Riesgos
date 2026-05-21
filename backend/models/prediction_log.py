from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from backend.database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    prediccion = Column(Float)
    features = Column(String)
    modelo = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)