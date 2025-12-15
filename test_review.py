import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Test a new review
sample_review = "This product is amazing and works perfectly!"
sample_vec = vectorizer.transform([sample_review])
prediction = model.predict(sample_vec)
print("Prediction (1=Real,0=Fake):", prediction[0])
