import pandas as pd

# Load your cleaned dataset
df = pd.read_csv("dataset/clean_reviews.csv")

# 1️⃣ review_text → from clean_text
df_new = pd.DataFrame()
df_new["review_text"] = df["clean_text"].astype(str)

# 2️⃣ label → based on Score
# Rule:
# Score >= 4 → Genuine (0)
# Score <= 2 → Fake (1)
# Remove Score == 3 (neutral)
df = df[df["Score"] != 3]

df_new = pd.DataFrame()
df_new["review_text"] = df["clean_text"].astype(str)

df_new["label"] = df["Score"].apply(
    lambda x: 0 if x >= 4 else 1
)

# 3️⃣ ocr_confidence → dummy value (since this is text data)
df_new["ocr_confidence"] = 0.90

# Save final dataset
df_new.to_csv("dataset/label_reviews.csv", index=False)

print("✅ reviews.csv created successfully!")
print(df_new.head())
