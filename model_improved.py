import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_advanced_model():

    # Load cleaned data
    df = pd.read_csv("dataset/clean_reviews.csv")

    # Label creation (if not present)
    if "Label" in df.columns:
        y = df['Label']
    else:
        y = df['Score'].apply(lambda x: 1 if x >= 3 else 0)

    X = df['clean_text'].astype(str)

    # Improved TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=7000,
        ngram_range=(1,2),         # bigrams improve accuracy
        min_df=3,
        max_df=0.9
    )

    X_vec = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # ---- MODELS ----
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Linear SVM": LinearSVC(),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.7,
            eval_metric="logloss"
        )
    }

    best_acc = 0
    best_model = None
    best_name = ""

    print("\n📌 Training Models...\n")

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    print("\n🔥 Best Model:", best_name)
    print("🔥 Best Accuracy:", best_acc)

    # Save best model
    pickle.dump(best_model, open("best_model.pkl", "wb"))
    pickle.dump(vectorizer, open("best_vectorizer.pkl", "wb"))

    print("\n✔ Saved: best_model.pkl & best_vectorizer.pkl")

    # Evaluation of best model
    preds = best_model.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, preds))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Save performance report
    with open("model_report.txt", "w") as f:
        f.write(f"BEST MODEL: {best_name}\n")
        f.write(f"ACCURACY: {best_acc}\n\n")
        f.write(classification_report(y_test, preds))

    print("\n✔ Performance report saved: model_report.txt")


train_advanced_model()
