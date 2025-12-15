# src/ocr_tuning_csv.py
import cv2
import pytesseract
import numpy as np
import os
import csv
from ocr import preprocess_image  # using your Day-9 preprocessing


# ---- Tesseract Path ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_tuned(path, psm=6, oem=1):
    """
    OCR with custom configurations: returns text, confidence, config_string
    """
    img = preprocess_image(path)

    config = f"--oem {oem} --psm {psm}"

    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

    # collect text
    words = [w for w in data["text"] if w.strip() != ""]
    text = " ".join(words)

    # collect confidence values
    confs = []
    for c in data["conf"]:
        c = str(c).strip()
        if c and c != "-1":
            try:
                confs.append(int(c))
            except:
                pass

    mean_conf = np.mean(confs) if confs else 0

    return text, mean_conf, config


def create_csv_log():
    """
    Runs OCR tuning on all images and saves results into CSV
    """
    images_folder = r"E:\fake-review-detector\data\screenshots"
    output_csv = r"E:\fake-review-detector\data\ocr_output\ocr_tuning_results.csv"

    # PSM and OEM values to test
    test_psm = [3, 6, 11, 13]
    test_oem = [1]

    # get all images
    images = [
        os.path.join(images_folder, f)
        for f in os.listdir(images_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    # write CSV header
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "psm", "oem", "confidence", "text"])

        # process each image with each config
        for img_path in images:  # you can test more if you want
            img_name = os.path.basename(img_path)
            print(f"\nProcessing: {img_name}")

            for psm in test_psm:
                for oem in test_oem:
                    text, conf, cfg = ocr_tuned(img_path, psm, oem)

                    # write row
                    writer.writerow([img_name, psm, oem, conf, text[:200]])

                    print(f"PSM={psm}, OEM={oem}, Conf={conf:.2f}")


if __name__ == "__main__":
    print("Running OCR tuning + saving CSV...")
    create_csv_log()
    print("\nCSV saved: data/ocr_output/ocr_tuning_results.csv")
