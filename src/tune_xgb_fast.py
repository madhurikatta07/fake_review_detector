import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("🔹 Loading data...")
df = pd.read_csv("dataset/label_reviews.csv")

X = df["review_text"].astype(str)
y = df["label"]

print("🔹 TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=3000,     # 🔥 reduced for speed
    stop_words="english"
)

X_vec = vectorizer.fit_transform(X)

print("🔹 Train/Test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

print("🚀 Training FAST model...")
model = LogisticRegression(
    max_iter=300,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("✅ Training done!")

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/mfast_model.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")

print("🎉 FAST MODEL SAVED")
