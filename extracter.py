import os
import cv2
import pytesseract
from deep_translator import GoogleTranslator
import re

# -------------------------
# Configure Tesseract
# -------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

# -------------------------
# Function to extract fields
# -------------------------
def extract_fields(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found at: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("❌ Could not load image. Check file format/path.")

    # Preprocess image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR Arabic text
    try:
        text = pytesseract.image_to_string(gray, lang='ara')
    except pytesseract.TesseractError as e:
        raise RuntimeError(f"❌ Tesseract OCR failed: {e}")

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Translate all lines to English
    translated_lines = []
    for line in lines:
        try:
            translated_lines.append(GoogleTranslator(source='ar', target='en').translate(line))
        except:
            translated_lines.append(line)

    # Initialize fields
    data = {
        "Patient Name": "Not found",
        "Date": "Not found",
        "Doctor": "Not found",
        "Hospital/Clinic": "Not found",
        "Notes": "Not found",
        "Medicines": []
    }

    # -------------------------
    # Heuristic extraction
    # -------------------------
    for line in translated_lines:
        low_line = line.lower()

        # Patient Name: first line containing "name" or likely a person's name
        if data["Patient Name"] == "Not found" and any(k in low_line for k in ["name", "patient"]):
            data["Patient Name"] = line
        # Date: lines containing numbers + month keywords
        elif data["Date"] == "Not found" and re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]?\d{0,4}', line):
            data["Date"] = line
        elif data["Date"] == "Not found" and any(month in low_line for month in ["jan", "feb", "mar", "apr", "may", "jun",
                                                                              "jul", "aug", "sep", "oct", "nov", "dec"]):
            data["Date"] = line
        # Doctor: contains 'dr' or 'doctor'
        elif data["Doctor"] == "Not found" and any(k in low_line for k in ["dr", "doctor"]):
            data["Doctor"] = line
        # Hospital/Clinic: contains 'hospital' or 'clinic'
        elif data["Hospital/Clinic"] == "Not found" and any(k in low_line for k in ["hospital", "clinic"]):
            data["Hospital/Clinic"] = line
        # Medicines: lines containing mg/ml or other medicine indicators
        elif re.search(r'\d+\s?(mg|ml)', line, re.IGNORECASE):
            data["Medicines"].append(line)
        # Notes: fallback, append remaining lines
        elif data["Notes"] == "Not found":
            data["Notes"] = line

    if not data["Medicines"]:
        data["Medicines"] = ["Not found"]

    return data

# -------------------------
# Main program
# -------------------------
if __name__ == "__main__":
    image_path = input("Enter the full path of the Arabic receipt image: ").strip().strip('"')

    try:
        result = extract_fields(image_path)
        print("\n✅ Extracted Information (English):")
        for key, value in result.items():
            if isinstance(value, list):
                value = [str(v) if v is not None else "Not found" for v in value]
                print(f"{key}: {', '.join(value)}")
            else:
                print(f"{key}: {value if value is not None else 'Not found'}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\nPress Enter to exit...")
