from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from backend.crud import create_emotion_prediction, get_predictions
from backend.database import SessionLocal, engine
from backend.schemas import InputText
from backend.pretrained_model import tokenizer, model
from backend.model import Base
import torch

Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/classify-emotion/")
async def classify_emotion(input_text: InputText, db: Session = Depends(get_db)):
    inputs = tokenizer(input_text.text, return_tensors="pt")

    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    prediction = create_emotion_prediction(db, input_text, probs)

    return {"input_text": input_text.text, "emotion_probs": probs[0].tolist()}

# API endpoint to retrieve stored predictions from the database
@app.get("/get-predictions/")
async def read_predictions(limit: int = 10, db: Session = Depends(get_db)):
    predictions = get_predictions(db, limit)
    return predictions
