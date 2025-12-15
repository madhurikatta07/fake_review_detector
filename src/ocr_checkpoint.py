# src/ocr_checkpoint.py
import csv
import os
import random

# --- Input & Output ---
input_csv = r"E:\fake-review-detector\data\ocr_output\ocr_output_clean.csv"
output_csv = r"E:\fake-review-detector\data\ocr_output\ocr_output_sample.csv"

# Create folder if not exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# --- Read all rows ---
with open(input_csv, "r", encoding="utf-8") as f_in:
    reader = list(csv.DictReader(f_in))
    fieldnames = reader[0].keys() if reader else []

    # Random sample 15 rows (or less if dataset smaller)
    sample_rows = random.sample(reader, min(15, len(reader)))

# --- Write sample CSV ---
with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    for row in sample_rows:
        writer.writerow(row)

print(f"\n✅ Sample OCR CSV saved: {output_csv}")
