import nltk
nltk.download('punkt')
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

# Load from Hugging Face repository instead of local path
repo_id = "SaiRohitMurali/distilbertmodel-598"

# Load tokenizer from the repository
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Load base model from the repository
base_model = AutoModelForSequenceClassification.from_pretrained(
    repo_id,
    num_labels=28
)

# Load PEFT configuration
peft_config = PeftConfig.from_pretrained(repo_id)

# Load the complete model with adapter
model = PeftModel.from_pretrained(base_model, repo_id)

def segment_text(text: str):
    segments = nltk.sent_tokenize(text)
    return segments

id2label = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label_id = torch.argmax(logits, dim=1).item()
        predicted_label = id2label[predicted_label_id]
        return predicted_label

def analyze_emotion_transitions(text: str):
    segments = segment_text(text)
    emotion_trace = []

    for segment in segments:
        emotion = predict(segment)
        emotion_trace.append((segment, emotion))

    return emotion_trace

# Try it out
prompt = "idk why i started yapping then my bf got really mad at me and wanted to break up with me. what do I even do now. I'm such a mess, gonna go get some wine and then drunk call all my friends"
# predictions = analyze_emotion_transitions(prompt)
predictions = predict(prompt)
print(predictions)
