from sqlalchemy.orm import Session
from .model import EmotionPrediction
from .schemas import InputText

def create_emotion_prediction(db: Session, input_text: InputText, probs):
    prediction = EmotionPrediction(
        input_text=input_text.text,
        anger_prob=probs[0][0].item(),
        fear_prob=probs[0][1].item(),
        joy_prob=probs[0][2].item(),
        sadness_prob=probs[0][3].item(),
        surprise_prob=probs[0][4].item(),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction

def get_predictions(db: Session, limit: int = 10):
    return db.query(EmotionPrediction).limit(limit).all()
