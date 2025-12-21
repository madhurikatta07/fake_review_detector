import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# ---------- LOAD DATA ----------
print("🔹 Loading data...")
df = pd.read_csv("dataset/label_reviews.csv")

X = df["review_text"].astype(str)
y = df["label"]

# ---------- LOAD MODEL & VECTORIZER ----------
print("🔹 Loading model...")
model = joblib.load("models/fast_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

X_vec = vectorizer.transform(X)

# ---------- PREDICTION ----------
y_pred = model.predict(X_vec)

print("\n📊 Classification Report (Full Dataset):")
print(classification_report(y, y_pred))

# ---------- CROSS VALIDATION ----------
print("\n🔁 Running Cross-Validation (F1-score)...")
scores = cross_val_score(
    model,
    X_vec,
    y,
    cv=5,
    scoring="f1"
)

print("F1-scores:", scores)
print("Average F1-score:", scores.mean())
print("\n✅ Evaluation complete.")