import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

def train_model():
    # Load cleaned dataset
    df = pd.read_csv("dataset/clean_reviews.csv")
    
    # Create labels if not present
    if 'Label' in df.columns:
        y = df['Label']
    else:
        y = df['Score'].apply(lambda x: 1 if x >= 3 else 0)
    
    # Clean text column
    X = df['clean_text'].dropna()
    X = X.astype(str)
    
    # Align y with X
    y = y[X.index]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    
    # Evaluation
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    
    # Save model & vectorizer
    pickle.dump(model, open("models/model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

    print("✔ Model and vectorizer saved as 'models/model.pkl' and 'models/vectorizer.pkl'")

train_model()


