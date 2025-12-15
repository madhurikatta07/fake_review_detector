Fake Review Detection System — Week 1
📅 Week 1 — Setup, Data Collection & Baseline (Text-only)

Goal: Set up the project environment, collect initial data, and create a baseline for fake review detection.

🔹 Day 0 — Initial Setup

Installed Python 3.9+ and Git.

Installed Tesseract OCR for text extraction from images.

Created project folder and virtual environment:

mkdir fake-review-detector
cd fake-review-detector
python -m venv venv


Activated virtual environment:

Mac/Linux: source venv/bin/activate

Windows: venv\Scripts\activate

Installed required Python packages (e.g., pandas, numpy, scikit-learn, matplotlib, seaborn, etc.).

🔹 Day 1 — Project Structure

Created main project folders:

fake-review-detector/
│
├─ data/           # Raw and processed review data
├─ src/            # Python scripts for processing, ML, OCR
├─ notebooks/      # Jupyter notebooks for experimentation
├─ models/         # Trained ML models
└─ README.md


Set up Git repository and .gitignore file:

git init
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore

🔹 Day 2 — Data Collection

Collected text-only review data from various sources (websites, e-commerce platforms, CSV files).

Performed basic data cleaning:

Removed duplicates

Removed empty reviews

Normalized text (lowercase, removed special characters)

Saved cleaned data as data/reviews_cleaned.csv.

🔹 Day 3 — Exploratory Data Analysis (EDA)

Analyzed the review dataset:

Count of reviews per product.

Average review length.

Distribution of ratings.

Visualized data using matplotlib and seaborn.

🔹 Day 4 — Baseline Text Classification

Implemented a baseline model for fake review detection:

Technique: TF-IDF + Logistic Regression

Split data into train/test sets (80/20 split)

Evaluated baseline performance (accuracy, precision, recall, F1-score)

Observed that baseline model provides a starting point for ML improvements.

🔹 Day 5 — Preliminary Insights

Generated a small report summarizing Week 1:

Dataset size: X reviews

Number of fake vs. real reviews: X/X

Initial baseline model accuracy: X%

Prepared the project for Week 2: moving from text-only baseline to OCR-based review extraction.

✅ Week 1 Summary

Environment setup completed.

Project folder structured and Git initialized.

Initial dataset collected and cleaned.

Basic exploratory data analysis performed.

Baseline fake review detection model implemented.