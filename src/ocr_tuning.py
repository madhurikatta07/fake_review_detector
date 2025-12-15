# src/ocr_tuning.py
import cv2
import pytesseract
import numpy as np
import os
from ocr import preprocess_image  # import your Day-9 function


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_tuned(path, psm=6, oem=1):
    """
    OCR with custom PSM & OEM settings
    """
    img = preprocess_image(path)

    config = f"--oem {oem} --psm {psm}"

    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

    words = [w for w in data["text"] if w.strip() != ""]
    text = " ".join(words)

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



if __name__ == "__main__":
    folder = r"E:\fake-review-detector\data\screenshots"
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".jpg", ".png"))]

    test_psm = [3, 6, 11, 13]
    test_oem = [1]

    print("\n========== OCR TUNING TEST ==========\n")

    for img_path in images[:10]:  # test on first 10 images
        print(f"\nImage: {os.path.basename(img_path)}")

        for psm in test_psm:
            for oem in test_oem:
                text, conf, cfg = ocr_tuned(img_path, psm, oem)
                print(f"Config = {cfg} | Conf = {conf:.2f}")
