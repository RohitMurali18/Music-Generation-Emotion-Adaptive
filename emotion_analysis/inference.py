# emotion_analysis/inference.py

import torch
from emotion_analysis.modeling import load_model
from emotion_analysis.config import ID2LABEL
from emotion_analysis.data_preprocessing import segment_text

tokenizer, model = load_model()

def predict(text: str) -> str:
    """
    Predict the emotion for a single text input.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label_id = torch.argmax(logits, dim=1).item()
        predicted_label = ID2LABEL[predicted_label_id]
    return predicted_label

def analyze_emotion_transitions(text: str):
    """
    Break text into segments and get an emotion label for each segment.
    """
    segments = segment_text(text)
    emotion_trace = []

    for segment in segments:
        emotion = predict(segment)
        emotion_trace.append((segment, emotion))

    return emotion_trace
