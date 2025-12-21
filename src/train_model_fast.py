import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("🔹 Loading dataset...")
df = pd.read_csv("dataset/label_reviews.csv")

X = df["review_text"].astype(str)
y = df["label"]

print("🔹 Vectorizing text (TF-IDF)...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_vec = vectorizer.fit_transform(X)

print("🔹 Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("🔹 Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/fast_model.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")

print("✅ FAST MODEL TRAINED & SAVED")
