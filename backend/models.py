from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from .db import Base

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(10), nullable=False)        # 'text' | 'url'
    source_value = Column(Text, nullable=True)         # url o fragmento de texto recortado
    label = Column(String(10), nullable=False)         # Fake/Real
    score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
