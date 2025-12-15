import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# Load cleaned dataset
df = pd.read_csv("dataset/clean_reviews.csv")

# 🔧 Fix NaN or float values in clean_text
df['clean_text'] = df['clean_text'].fillna("").astype(str)

# 🔥 1) Dataset Info
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# If no Label column exists, create it from Score
if 'Label' not in df.columns:
    df['Label'] = df['Score'].apply(lambda x: 1 if x >= 3 else 0)

# 🔥 2) Review length
df['review_length'] = df['clean_text'].apply(lambda x: len(str(x).split()))

# 🔥 3) Combined Plot: Class Distribution + Review Length
plt.figure(figsize=(14, 6))

# Left plot — Fake vs Real
plt.subplot(1, 2, 1)
df['Label'].value_counts().plot(kind='bar')
plt.title("Fake vs Real Review Distribution (1=Real, 0=Fake)")
plt.xlabel("Label")
plt.ylabel("Count")

# Right plot — Review length histogram
plt.subplot(1, 2, 2)
sns.histplot(df['review_length'], bins=50)
plt.title("Review Length Distribution")
plt.xlabel("Number of words")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# 🔥 4) Most common words
all_words = " ".join(df['clean_text']).split()
counter = Counter(all_words)
print("\nMost common words:", counter.most_common(20))

# 🔥 5) WordClouds for Fake vs Real reviews
fake_text = " ".join(df[df['Label'] == 0]['clean_text'])
real_text = " ".join(df[df['Label'] == 1]['clean_text'])

wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
wordcloud_real = WordCloud(width=800, height=400, background_color='white').generate(real_text)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title("Fake Reviews WordCloud")

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title("Real Reviews WordCloud")

plt.show()
