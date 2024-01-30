from sqlalchemy.orm import Session
from .model import EmotionPrediction
from .schemas import InputText

EMOTION_CLASSES = ["anger", "fear", "joy", "sadness", "surprise", "neutral", "disgust"]

def create_emotion_prediction(db: Session, input_text: InputText, probs):
    prediction = EmotionPrediction(
        input_text=input_text.text,
        anger_prob_m=probs[0][0].item(),
        fear_prob_m=probs[0][1].item(),
        joy_prob_m=probs[0][2].item(),
        sadness_prob_m=probs[0][3].item(),
        surprise_prob_m=probs[0][4].item(),
        neutral_prob_m=probs[0][5].item(),
        disgust_prob_m=probs[0][6].item(),
        anger_prob_f=probs[0][7].item(),
        fear_prob_f=probs[0][8].item(),
        joy_prob_f=probs[0][9].item(),
        sadness_prob_f=probs[0][10].item(),
        surprise_prob_f=probs[0][11].item(),
        neutral_prob_f=probs[0][12].item(),
        disgust_prob_f=probs[0][13].item(),        
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction

def get_predictions(db: Session, limit: int = 10):
    return db.query(EmotionPrediction).limit(limit).all()
