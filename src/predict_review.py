import joblib
from ocr import extract_text_from_image

# ---------- LOAD MODEL ----------
print("🔹 Loading model & vectorizer...")
model = joblib.load("models/fast_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

# ---------- PREDICT FUNCTION ----------
def predict_review(review_text):
    review_text = str(review_text)
    X = vectorizer.transform([review_text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()

    label = "FAKE REVIEW ❌" if pred == 1 else "GENUINE REVIEW ✅"
    return label, round(prob, 2)


# ---------- TEST ----------
if __name__ == "__main__":
    
    print("Choose input type:")
    print("1. Text")
    print("2. Image")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        print("\nEnter a review text:")
        text = input(">>> ")
        
    elif choice == "2":
        print("\nEnter the image file path:")
        image_path = input(">>> ").strip()
        print("\nExtracting text from image...")
        text = extract_text_from_image(image_path)
        print("\nExtracted Text:\n", text)
    
    label, confidence = predict_review(text)
    print("\nResult:", label)
    print("Confidence:", confidence)
