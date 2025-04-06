# emotion_analysis/data_preprocessing.py

import nltk

def segment_text(text: str):  # this is a very basic implementation and i think its trash
    """
    Break text into sentences.
    """
    nltk.download("punkt", quiet=True)
    segments = nltk.sent_tokenize(text)
    return segments
