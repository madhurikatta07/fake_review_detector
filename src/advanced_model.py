import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# ---------- LOAD DATA ----------
df = pd.read_csv("dataset/reviews.csv")

print("Columns found:", df.columns)

# ---------- RENAME REQUIRED COLUMNS ----------
df = df.rename(columns={
    "Text": "review_text",
    "Score": "rating"
})

# ---------- CREATE LABEL ----------
# rating >= 4 → Genuine (1)
# rating <= 2 → Fake (0)
df = df[df["rating"].isin([1, 2, 4, 5])]
df["label"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

# ---------- TEXT + LABEL ----------
texts = df["review_text"].fillna("").astype(str)
labels = df["label"]

# ---------- TF-IDF (FAST) ----------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(texts)

# ---------- TRAIN TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ---------- XGBOOST (FAST MODE) ----------
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",   # 🚀 VERY FAST
    random_state=42
)

# ---------- TRAIN ----------
print("🚀 Training started...")
model.fit(X_train, y_train)
print("✅ Training finished")

# ---------- EVALUATION ----------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ---------- SAVE ----------
joblib.dump(model, "models/mode_text_xgb.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")

print("✅ Model & vectorizer saved successfully")
