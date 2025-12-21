# src/ocr.py
import cv2
import pytesseract
import numpy as np
from PIL import Image
import os

# ---- Tesseract Windows Path ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_image(image_path):
    """
    Extracts review text from image using OCR
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve OCR quality
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(gray)
    return text.strip()


def preprocess_image(path):
    """
    Takes image path -> returns cleaned, thresholded image
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Failed to load image file: {path}")

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # denoise
    gray = cv2.medianBlur(gray, 3)

    # increase contrast
    gray = cv2.equalizeHist(gray)

    # threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th




def ocr_from_image(path, lang='eng'):
    """
    Returns -> extracted_text, mean_confidence
    """

    img = preprocess_image(path)

    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

    # collect text
    words = [str(w).strip() for w in data["text"] if str(w).strip() != ""]
    text = " ".join(words)

    # ---- FIXED CONFIDENCE EXTRACTION ----
    confs = []
    for c in data["conf"]:
        c = str(c).strip()            # convert to string safely
        if c and c != "-1":           # ignore empty or -1 values
            try:
                confs.append(int(c))
            except:
                pass  # skip unknown formatting

    mean_conf = np.mean(confs) if confs else 0

    return text, mean_conf


if __name__ == "__main__":
    path = r"E:\fake-review-detector\data\screenshots\WhatsApp Image 2025-12-08 at 18.01.50_a9b1b7e0.jpg"
    print("Reading:", path)

    txt, conf = ocr_from_image(path)

    print("\nConfidence:", conf)
    print("\nExtracted Text:\n", txt[:300])





