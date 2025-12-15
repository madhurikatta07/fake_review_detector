import os
import csv
from ocr import ocr_from_image
from ocr_parser import extract_rating, extract_date, extract_username

# Folder containing all images
image_folder = "../data/screenshots/"  # Adjust if needed
output_csv = "../data/ocr_output/ocr_results.csv"  # CSV will be saved here

# List all image files (png, jpg, jpeg)
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not image_files:
    print("No images found in folder:", image_folder)
else:
    results = []

    for img_file in image_files:
        image_path = os.path.join(image_folder, img_file)

        # Check if file exists
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue

        # OCR
        try:
            text, conf = ocr_from_image(image_path)
        except Exception as e:
            print(f"Error OCR for {img_file}: {e}")
            continue

        # Extract info
        rating = extract_rating(text)
        date = extract_date(text)
        username = extract_username(text)

        # Append results
        results.append({
            "image_name": img_file,
            "username": username,
            "rating": rating,
            "date": date,
            "review_text": text
        })

        # Print results
        print(f"\n--- {img_file} ---")
        print("OCR Confidence:", conf)
        print("Username:", username)
        print("Rating:", rating)
        print("Date:", date)
        print("Review Text:", text[:200], "...\n")

    # Save to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["image_name", "username", "rating", "date", "review_text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n✅ All results saved to CSV: {output_csv}")
