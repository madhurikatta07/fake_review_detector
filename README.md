Fake Review Detection System
🔹 Project Overview

A machine learning-based system to detect fake product reviews from text and images.
The system uses a text-only baseline and an OCR pipeline for extracting review text from screenshots or photos to train ML models for real-world detection.

📅 Week 1 — Setup & Baseline (Text-only)

Goal: Set up project environment, collect initial data, and create a baseline model.

Installed Python, Git, and Tesseract OCR.

Created project structure and virtual environment.

Collected and cleaned text-only review data.

Performed exploratory data analysis (EDA).

Built a baseline ML model (TF-IDF + Logistic Regression).

Generated a preliminary report on dataset and model performance.

📅 Week 2 — OCR Pipeline & Image Preprocessing

Goal: Extract review text from screenshots/photos for ML input.

Collected 200–500 raw and synthetic review images.

Implemented preprocessing functions (grayscale, denoise, threshold) in src/ocr.py.

Integrated Tesseract OCR for text extraction.

Cleaned and normalized extracted text.

Structured data into CSV: review_id | product_id | review_text | source_image.

Validated OCR output and prepared checkpoint CSV for ML training.

✅ Next Steps

Week 3 onward: Train ML models using structured OCR data.

Evaluate, tune, and deploy a system for real-world fake review detection.