
from sqlalchemy import Column, Integer, String, Float

from .database import Base

# Define the database model
class EmotionPrediction(Base):
    __tablename__ = "emotion_predictions"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, index=True)
    anger_prob = Column(Float)
    fear_prob = Column(Float)
    joy_prob = Column(Float)
    sadness_prob = Column(Float)
    surprise_prob = Column(Float)
