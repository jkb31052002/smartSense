from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("michellejieli/emotion_text_classifier")
model = AutoModelForSequenceClassification.from_pretrained("michellejieli/emotion_text_classifier")

input_text = "null"

inputs = tokenizer(input_text, return_tensors="pt")

# Forward pass through the model
outputs = model(**inputs)

# Apply softmax to get probabilities
probs = torch.nn.functional.softmax(outputs.logits, dim=1)