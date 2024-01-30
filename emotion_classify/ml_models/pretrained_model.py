from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("michellejieli/emotion_text_classifier")
model = AutoModelForSequenceClassification.from_pretrained("michellejieli/emotion_text_classifier")

input_text = "null"

inputs = tokenizer(input_text, return_tensors="pt")

outputs = model(**inputs)

probs = torch.nn.functional.softmax(outputs.logits, dim=1)