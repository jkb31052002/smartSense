
from sqlalchemy import Column, Integer, String, Float

from .database import Base

# Define the database model
class EmotionPrediction(Base):
    __tablename__ = "emotion_predictions"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, index=True)
    anger_prob_m = Column(Float)
    fear_prob_m = Column(Float)
    joy_prob_m = Column(Float)
    sadness_prob_m = Column(Float)
    surprise_prob_m = Column(Float)
    neutral_prob_m = Column(Float)
    disgust_prob_m = Column(Float)
    anger_prob_r = Column(Float)
    disgust_prob_r = Column(Float)
    fear_prob_r = Column(Float)
    joy_prob_r = Column(Float)
    neutral_prob_r = Column(Float)
    sadness_prob_r = Column(Float)
    surprise_prob_r = Column(Float)



