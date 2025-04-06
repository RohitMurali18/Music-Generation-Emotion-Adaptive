# emotion_analysis/inference.py

import torch
from emotion_analysis.modeling import load_model
from emotion_analysis.config import ID2LABEL
from emotion_analysis.data_preprocessing import segment_text
import torch.nn.functional as F


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



def predict_all_labels(text: str) -> dict:
    """
    Predict scores for all emotion labels for a single text input.
    Returns a dictionary of label: score.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
    
    label_scores = {ID2LABEL[i]: round(prob, 4) for i, prob in enumerate(probabilities)}
    return label_scores


def predict_top_k_labels(text: str, k: int = 3) -> list:
    """
    Predict the top-k emotion labels with their scores for a single text input.
    Returns a list of (label, score) tuples sorted by score in descending order.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze()

    topk = torch.topk(probabilities, k)
    top_labels = []

    for idx, prob in zip(topk.indices, topk.values):
        label = ID2LABEL[idx.item()]
        score = round(prob.item(), 4)
        top_labels.append((label, score))

    return top_labels

def predict_labels_above_threshold(text: str, threshold: float = 0.2) -> list:
    """
    Predict emotion labels with probabilities greater than the given threshold.
    Returns a list of (label, score) tuples.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze()

    selected_labels = []
    for i, prob in enumerate(probabilities):
        score = prob.item()
        if score > threshold:
            label = ID2LABEL[i]
            selected_labels.append((label, round(score, 4)))

    return selected_labels


def analyze_emotion_transitions(text: str):    #not done yet
    """
    Break text into segments and get an emotion label for each segment.
    """
    segments = segment_text(text)
    emotion_trace = []

    for segment in segments:
        emotion = predict(segment)
        emotion_trace.append((segment, emotion))

    return emotion_trace
