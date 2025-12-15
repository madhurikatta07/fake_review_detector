import os
from ocr import ocr_from_image
from ocr_parser import extract_rating, extract_date, extract_username

# ✅ Folder containing all images
image_folder = "../data/screenshots/"  # Adjust path if needed

# List all image files (png, jpg, jpeg)
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not image_files:
    print("No images found in folder:", image_folder)
else:
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

        # Print results
        print(f"\n--- {img_file} ---")
        print("OCR Confidence:", conf)
        print("Username:", username)
        print("Rating:", rating)
        print("Date:", date)
        print("Review Text:", text[:200], "...\n")
