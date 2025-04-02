# emotion_analysis/modeling.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from emotion_analysis.config import REPO_ID, NUM_LABELS

def load_model():
    """
    Load the base model, tokenizer, and peft adapter.
    Return the tokenizer and the complete model object.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        REPO_ID, 
        num_labels=NUM_LABELS
    )
    peft_config = PeftConfig.from_pretrained(REPO_ID)

    model = PeftModel.from_pretrained(base_model, REPO_ID)

    model.eval()

    return tokenizer, model
