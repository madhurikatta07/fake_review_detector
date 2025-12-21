
import re
import numpy as np
import emoji
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer


sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- GLOBAL OBJECTS ----------
analyzer = SentimentIntensityAnalyzer()

# ---------- SIMPLE TEXT FEATURES ----------
def simple_text_features(text, ocr_conf=1.0):
    text = text if isinstance(text, str) else ""

    features = {
        "text_len": len(text),
        "num_exclaim": text.count("!"),
        "num_question": text.count("?"),
        "num_urls": len(re.findall(r"http\S+|www\S+", text)),
        "num_emojis": len([c for c in text if c in emoji.EMOJI_DATA]),
        "ocr_confidence": ocr_conf
    }

    sentiment = analyzer.polarity_scores(text)
    features["sentiment_pos"] = sentiment["pos"]
    features["sentiment_neg"] = sentiment["neg"]
    features["sentiment_neu"] = sentiment["neu"]
    features["sentiment_compound"] = sentiment["compound"]

    return features


# ---------- TF-IDF ----------
def fit_tfidf(texts, max_features=3000):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    return X


def transform_tfidf(texts):
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return vectorizer.transform(texts)


def get_sentence_embeddings(texts):
    """
    texts: list of review strings
    returns: numpy array of embeddings
    """
    texts = [t if isinstance(t, str) else "" for t in texts]
    embeddings = sbert_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def sentiment_score(text):
    return 0.0
