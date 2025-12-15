import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if isinstance(text, float):
        text = str(text)

    text = text.lower()  # lowercase
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # remove stopwords
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])  # lemmatization
    return text


def preprocess_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['Text'])

    df['clean_text'] = df['Text'].apply(clean_text)
    df.to_csv("dataset/clean_reviews.csv", index=False)

    print("✔ Preprocessing completed and saved as clean_reviews.csv")
    return df
