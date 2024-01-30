from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer_roberta = AutoTokenizer.from_pretrained("badmatr11x/roberta-base-emotions-detection-from-text")
model_roberta = AutoModelForSequenceClassification.from_pretrained("badmatr11x/roberta-base-emotions-detection-from-text")
