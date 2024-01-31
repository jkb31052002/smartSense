from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from backend.crud import create_emotion_prediction, get_predictions
from backend.database import SessionLocal, engine
from backend.schemas import InputText
from ml_models.pretrained_model import tokenizer, model
from ml_models.pretrained_model_roberta import tokenizer_roberta, model_roberta
from backend.model import Base, EmotionPrediction
import torch.nn.functional as F

Base.metadata.create_all(bind=engine)

EMOTION_CLASSES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_emotion_prediction(db: Session, input_text: InputText, probs_m, probs_r):
    prediction = EmotionPrediction(input_text=input_text.text)

    for i, (emotion_prob_m, emotion_prob_r) in enumerate(zip(probs_m[0], probs_r[0])):
        setattr(prediction, f"{EMOTION_CLASSES[i]}_prob_m", emotion_prob_m.item())
        setattr(prediction, f"{EMOTION_CLASSES[i]}_prob_r", emotion_prob_r.item())

    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction

@app.post("/classify-emotion/")
async def classify_emotion(input_text: InputText, db: Session = Depends(get_db)):
    inputs_m = tokenizer(input_text.text, return_tensors="pt")
    outputs_m = model(**inputs_m)
    probs_m = F.softmax(outputs_m.logits, dim=1)

    inputs_r = tokenizer_roberta(input_text.text, return_tensors="pt")
    outputs_r = model_roberta(**inputs_r)
    probs_r = F.softmax(outputs_r.logits, dim=1)

    prediction = create_emotion_prediction(db, input_text, probs_m, probs_r)

    return {"input_text": input_text.text, "emotion_prob_m": probs_m[0].tolist(), "emotion_prob_r": probs_r[0].tolist()}

@app.get("/get-predictions/")
async def read_predictions(limit: int = 10, db: Session = Depends(get_db)):
    predictions = get_predictions(db, limit)
    return predictions
