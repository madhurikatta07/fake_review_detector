# src/ocr_batch_csv.py
import os
import csv
from ocr import ocr_from_image
from ocr_parser import extract_rating, extract_date, extract_username

# --- Folders ---
images_folder = r"E:\fake-review-detector\data\screenshots"
output_csv = r"E:\fake-review-detector\data\ocr_output\ocr_output.csv"

# Get all images
images = [os.path.join(images_folder, f) 
          for f in os.listdir(images_folder) 
          if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Create CSV folder if not exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Write CSV header
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "ocr_confidence", "username", "rating", "date", "text"])

    # Process all images
    for img_path in images:
        img_name = os.path.basename(img_path)
        print(f"Processing: {img_name}")

        try:
            # Run OCR
            text, conf = ocr_from_image(img_path)

            # Parse structured info
            username = extract_username(text)
            rating = extract_rating(text)
            date = extract_date(text)

            # Write row
            writer.writerow([img_name, conf, username, rating, date, text[:500]])  # store first 500 chars

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
