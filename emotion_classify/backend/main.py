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

EMOTION_CLASSES = ["anger", "fear", "joy", "sadness", "surprise", "neutral", "disgust"]

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_emotion_prediction(db: Session, input_text: InputText, probs_m, probs_f):
    prediction = EmotionPrediction(input_text=input_text.text)

    for i, (emotion_prob_m, emotion_prob_f) in enumerate(zip(probs_m[0], probs_f[0])):
        setattr(prediction, f"{EMOTION_CLASSES[i]}_prob_m", emotion_prob_m.item())
        setattr(prediction, f"{EMOTION_CLASSES[i]}_prob_f", emotion_prob_f.item())

    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction

@app.post("/classify-emotion/")
async def classify_emotion(input_text: InputText, db: Session = Depends(get_db)):
    inputs_m = tokenizer(input_text.text, return_tensors="pt")
    outputs_m = model(**inputs_m)
    probs_m = F.softmax(outputs_m.logits, dim=1)

    inputs_f = tokenizer_roberta(input_text.text, return_tensors="pt")
    outputs_f = model_roberta(**inputs_f)
    probs_f = F.softmax(outputs_f.logits, dim=1)

    prediction = create_emotion_prediction(db, input_text, probs_m, probs_f)

    return {"input_text": input_text.text, "emotion_probs_m": probs_m[0].tolist(), "emotion_probs_f": probs_f[0].tolist()}

# @app.post("/classify-emotion-michel/")
# async def classify_emotion(input_text: InputText, db: Session = Depends(get_db)):
#     inputs = tokenizer(input_text.text, return_tensors="pt")

#     outputs = model(**inputs)

#     probs = torch.nn.functional.softmax(outputs.logits, dim=1)

#     prediction = create_emotion_prediction(db, input_text, probs)

#     return {"input_text": input_text.text, "emotion_probs": probs[0].tolist()}

# async def classify_emotion_roberta(input_text: InputText, db: Session = Depends(get_db)):
#     inputs_r = tokenizer_roberta(input_text.text, return_tensors="pt")

#     outputs_r = model_roberta(**inputs_r)

#     probs_r = torch.nn.functional.softmax(input_text.text, return_tensors="pt")

#     prediction = create_emotion_prediction(db, input_text, probs_r)

#     return {"input_text": input_text.text, "emotion_probs": probs_r[9].tolist()}


@app.get("/get-predictions/")
async def read_predictions(limit: int = 10, db: Session = Depends(get_db)):
    predictions = get_predictions(db, limit)
    return predictions
