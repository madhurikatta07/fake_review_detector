from features import get_sentence_embeddings

reviews = [
    "Amazing product! Totally worth it",
    "Worst purchase of my life",
    "Very good quality and fast delivery"
]

embeddings = get_sentence_embeddings(reviews)

print("Embedding shape:", embeddings.shape)
print("First review vector (first 5 values):")
print(embeddings[0][:5])
