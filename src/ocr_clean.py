# src/ocr_clean.py
import csv
import re
import os

# --- Input & Output ---
input_csv = r"E:\fake-review-detector\data\ocr_output\ocr_output.csv"
output_csv = r"E:\fake-review-detector\data\ocr_output\ocr_output_clean.csv"

# Create folder if not exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)


def clean_text(text):
    """
    Post-OCR cleaning:
    - remove duplicated whitespace
    - fix common OCR errors
    - remove unwanted characters
    """
    if not text:
        return ""

    # replace common OCR mistakes
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("I0", "10").replace("l", "I")  # example fix
    text = text.replace("0", "O")  # optional, depends on your dataset

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --- Process CSV ---
with open(input_csv, "r", encoding="utf-8") as f_in, \
     open(output_csv, "w", newline="", encoding="utf-8") as f_out:

    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        row["text"] = clean_text(row["text"])

        # optionally, clean username
        if row["username"]:
            row["username"] = clean_text(row["username"])

        # optionally, standardize rating as float
        if row["rating"]:
            try:
                row["rating"] = float(row["rating"])
            except:
                row["rating"] = None

        # write cleaned row
        writer.writerow(row)

print(f"\n✅ Cleaned OCR CSV saved: {output_csv}")
