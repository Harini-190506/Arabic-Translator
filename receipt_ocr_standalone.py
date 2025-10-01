import sys
import os
import re
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
from PIL import Image, ImageOps

try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None  # type: ignore

try:
    import pytesseract  # type: ignore
    # Windows fallback path if not on PATH
    if sys.platform.startswith('win') and not os.environ.get('TESSERACT_CMD'):
        tcmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        if os.path.exists(tcmd):
            pytesseract.pytesseract.tesseract_cmd = tcmd
except Exception:
    pytesseract = None  # type: ignore

try:
    from deep_translator import GoogleTranslator  # type: ignore
except Exception:
    GoogleTranslator = None  # type: ignore

def is_arabic_text(s: str) -> bool:
    return any('\u0600' <= ch <= '\u06FF' for ch in (s or ''))

def arabic_ratio(s: str) -> float:
    if not s:
        return 0.0
    total = len([c for c in s if not c.isspace()])
    if total == 0:
        return 0.0
    arabic = sum(1 for c in s if '\u0600' <= c <= '\u06FF')
    return arabic / total

def should_translate(text: str) -> bool:
    if not text:
        return False
    # Do not translate if it's mostly numbers/symbols or already Latin
    if sum(ch.isdigit() for ch in text) / max(1, len(text)) > 0.6:
        return False
    if arabic_ratio(text) >= 0.2:  # at least 20% Arabic characters
        return True
    return False

def translate_to_english(text: str) -> str:
    if not text or GoogleTranslator is None:
        return text
    try:
        # Translate line by line to avoid mangling codes and amounts
        lines = text.split('\n')
        out_lines = []
        for ln in lines:
            if should_translate(ln):
                out_lines.append(GoogleTranslator(source='ar', target='en').translate(ln))
            else:
                out_lines.append(ln)
        return '\n'.join(out_lines)
    except Exception:
        return text


def read_image_oriented(path: str) -> np.ndarray:
    """Read an image and apply EXIF orientation if present. Returns BGR np array."""
    pil = Image.open(path)
    try:
        pil = ImageOps.exif_transpose(pil)
    except Exception:
        pass
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def deskew_image(image_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Deskew image using minAreaRect; returns rotated image and angle used."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    den = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    thr = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr == 0))  # text as dark
    angle = 0.0
    if coords.size > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
    if abs(angle) < 0.5:
        return image_bgr, 0.0
    (h, w) = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def preprocess(image_bgr: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return primary cleaned image and variants for OCR."""
    # Resize height to ~1600px to improve OCR resolution
    target_h = 1600
    if image_bgr.shape[0] < target_h:
        scale = target_h / float(image_bgr.shape[0])
        image_bgr = cv2.resize(image_bgr, (int(image_bgr.shape[1] * scale), target_h))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    den = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)

    # Unsharp mask (sharpen)
    gblur = cv2.GaussianBlur(den, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(den, 1.5, gblur, -0.5, 0)

    # Adaptive and Otsu thresholds
    adaptive = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Morphological cleanup
    cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))

    # Horizontal morphology to help line grouping (not mandatory for OCR but for visuals)
    horiz = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))

    variants = [cleaned, otsu, adaptive, sharp, horiz]
    return variants[0], variants


def easyocr_recognize(image: np.ndarray) -> Tuple[List[Tuple[List[Tuple[int, int]], str, float]], float]:
    if easyocr is None:
        return [], -1.0
    reader = easyocr.Reader(['en', 'ar'], gpu=False)
    result = reader.readtext(image)
    if not result:
        return [], -1.0
    avg_conf = float(np.mean([r[2] for r in result]))
    return result, avg_conf


def tesseract_recognize(image: np.ndarray) -> Tuple[List[Tuple[List[Tuple[int, int]], str, float]], float]:
    if pytesseract is None:
        return [], -1.0
    data = pytesseract.image_to_data(image, lang='ara+eng', config='--oem 1 --psm 6', output_type=pytesseract.Output.DICT)
    n = len(data.get('text', []))
    results = []
    confs = []
    for i in range(n):
        txt = (data['text'][i] or '').strip()
        if not txt:
            continue
        x, y, w, h = int(data['left'][i]), int(data['top'][i]), int(data['width'][i]), int(data['height'][i])
        conf = float(data['conf'][i]) if data['conf'][i] not in ('-1', '') else -1.0
        poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        results.append((poly, txt, conf))
        if conf >= 0:
            confs.append(conf)
    avg_conf = float(np.mean(confs)) if confs else -1.0
    return results, avg_conf


def ocr_pipeline(image_bgr: np.ndarray) -> Tuple[List[Tuple[List[Tuple[int, int]], str, float]], np.ndarray]:
    deskewed, _ = deskew_image(image_bgr)
    primary, variants = preprocess(deskewed)

    # Try EasyOCR first
    best_res: List[Tuple[List[Tuple[int, int]], str, float]] = []
    best_conf = -1.0
    if easyocr is not None:
        res, conf = easyocr_recognize(primary)
        best_res, best_conf = res, conf

    # Fallback or improvement via Tesseract
    for v in variants:
        res_t, conf_t = tesseract_recognize(v)
        if conf_t > best_conf:
            best_res, best_conf = res_t, conf_t

    # If still nothing, attempt EasyOCR on otsu variant
    if not best_res and easyocr is not None:
        res2, conf2 = easyocr_recognize(variants[1])
        if conf2 > best_conf:
            best_res, best_conf = res2, conf2

    return best_res, primary


def normalize_digits(text: str) -> str:
    return text.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789'))


def parse_fields(results: List[Tuple[List[Tuple[int, int]], str, float]], image_h: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Convert results to unified list of boxes
    boxes: List[Dict[str, Any]] = []
    for poly, txt, conf in results:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': normalize_digits(txt), 'conf': float(conf)})

    # Heuristic bands by y center
    max_y = max((b['y'] + b['h']) for b in boxes) if boxes else image_h
    top_band = max_y * 0.2
    bot_band = max_y * 0.8

    header = [b for b in boxes if b['y'] <= top_band]
    middle = [b for b in boxes if top_band < b['y'] < bot_band]
    bottom = [b for b in boxes if b['y'] >= bot_band]

    lines_sorted = sorted(boxes, key=lambda b: (b['y'], b['x']))
    all_text = '\n'.join([b['text'] for b in lines_sorted])

    # Regexes
    date_re = re.compile(r"(\b\d{1,2}[\-/]\d{1,2}[\-/]\d{4}\b|\b\d{4}[\-/]\d{1,2}[\-/]\d{1,2}\b|\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b)")
    total_kw = ["Total", "Grand Total", "Amount Due", "Balance Due", "Net Payable", "الإجمالي", "المجموع"]

    fields: Dict[str, Any] = {
        'patient_name': '',
        'age': '',
        'gender': '',
        'date': '',
        'doctor': '',
        'hospital': '',
        'medicines': [],
        'dosage': [],
        'amount': '',
        'notes': ''
    }

    # Hospital: top header words
    if header:
        header_text = ' '.join([b['text'] for b in header[:20]])
        fields['hospital'] = header_text[:120]

    # Patient name: top lines mentioning Name keywords, else first 2+ word line
    for b in header[:30]:
        if re.search(r"\b(Name|Patient Name|اسم\s*المريض|الاسم)\b", b['text'], re.IGNORECASE):
            t = re.sub(r"^.*?:", "", b['text']).strip()
            if len(t.split()) >= 1:
                fields['patient_name'] = t
                break
    if not fields['patient_name']:
        for b in header[:30]:
            if len(b['text'].split()) >= 2 and not re.search(r"\b(Dr|Doctor|Hospital|Clinic)\b", b['text'], re.IGNORECASE):
                fields['patient_name'] = b['text']
                break

    # Doctor: any line with Dr/Doctor
    for b in boxes:
        if re.search(r"\b(Dr\.?|Doctor|دكتور)\b", b['text'], re.IGNORECASE):
            fields['doctor'] = b['text']
            break

    # Age/Gender patterns
    for b in boxes:
        txt = b['text']
        m_age = re.search(r"\b(Age|العمر)\s*[:：-]?\s*(\d{1,3})\b", txt, re.IGNORECASE)
        if m_age and not fields['age']:
            fields['age'] = m_age.group(2)
        m_gender = re.search(r"\b(Gender|Sex|الجنس)\s*[:：-]?\s*(Male|Female|M|F|ذكر|أنثى)\b", txt, re.IGNORECASE)
        if m_gender and not fields['gender']:
            fields['gender'] = m_gender.group(2)

    # Date: first valid occurrence
    m = date_re.search(all_text)
    if m:
        fields['date'] = m.group(0)

    # Amount/Total: scan bottom then all lines for known keywords + amount
    def find_amount(candidates: List[Dict[str, Any]]) -> str:
        for b in candidates:
            line = b['text']
            if any(k.lower() in line.lower() for k in total_kw):
                m2 = re.search(r"(?:₹|\$|SAR)?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})\b", line)
                if m2:
                    return f"{line.split(':')[0]}: {m2.group(0)}"
        return ''

    fields['amount'] = find_amount(bottom)
    if not fields['amount']:
        fields['amount'] = find_amount(lines_sorted[-40:])

    # Medicines & Dosage: middle lines that have words and optional price, capture dosage tokens
    meds = []
    dosages = []
    price_re = re.compile(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})\b")
    dose_re = re.compile(r"\b(\d+\s*(mg|ml|mcg)|tablet[s]?|tab\b|caps?\b|قرص|كبسولة|مل|مجم)\b", re.IGNORECASE)
    for b in middle:
        t = b['text']
        if re.search(r"\b(total|amount|balance|vat|tax)\b", t, re.IGNORECASE):
            continue
        if any(ch.isalpha() for ch in t) and (price_re.search(t) or len(t.split()) >= 2):
            meds.append(t)
        m_d = dose_re.search(t)
        if m_d:
            dosages.append(t)
    if meds:
        fields['medicines'] = meds[:40]
    if dosages:
        fields['dosage'] = list(dict.fromkeys(dosages))[:30]

    # Notes: whatever bottom not used
    notes = []
    for b in bottom:
        if b['text'] not in fields['medicines'] and b['text'] not in fields['amount']:
            notes.append(b['text'])
    if notes:
        fields['notes'] = ' | '.join(notes[:10])

    # Translate Arabic content to English for display (conservative)
    for key in ['patient_name', 'doctor', 'hospital']:
        fields[key] = translate_to_english(fields.get(key, ''))
    # Keep date/amount as-is to avoid localization issues
    fields['medicines'] = [translate_to_english(x) for x in (fields.get('medicines') or [])]
    fields['dosage'] = [translate_to_english(x) for x in (fields.get('dosage') or [])]
    # Only translate notes if contains Arabic
    notes_val = fields.get('notes', '')
    fields['notes'] = translate_to_english(notes_val) if should_translate(notes_val) else notes_val

    return fields, boxes


def annotate(image: np.ndarray, boxes: List[Dict[str, Any]], fields: Dict[str, Any]) -> np.ndarray:
    vis = image.copy()
    for b in boxes:
        x, y, w, h = int(b['x']), int(b['y']), int(b['w']), int(b['h'])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 80), 2)
    # Highlight important fields by re-searching their text in boxes
    key_map = {
        'patient_name': (255, 140, 0),
        'age': (100, 255, 255),
        'gender': (255, 100, 255),
        'doctor': (0, 140, 255),
        'hospital': (180, 0, 255),
        'date': (255, 0, 0),
        'amount': (0, 255, 0)
    }
    for key, color in key_map.items():
        val = fields.get(key) or ''
        if isinstance(val, str) and val:
            for b in boxes:
                if val.lower() in b['text'].lower():
                    x, y, w, h = int(b['x']), int(b['y']), int(b['w']), int(b['h'])
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
                    break
    # Draw medicines/dosage lines
    for lst, color in [(fields.get('medicines') or [], (0, 200, 80)), (fields.get('dosage') or [], (200, 0, 120))]:
        for val in lst[:20]:
            for b in boxes:
                if val.lower() in b['text'].lower():
                    x, y, w, h = int(b['x']), int(b['y']), int(b['w']), int(b['h'])
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                    break
    return vis


def main():
    if len(sys.argv) < 2:
        print("Usage: python receipt_ocr_standalone.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    image = read_image_oriented(image_path)
    results, proc = ocr_pipeline(image)
    fields, boxes = parse_fields(results, proc.shape[0])

    # Print results
    print("\n=== Extracted Fields ===")
    print(f"Patient Name: {fields.get('patient_name','')}")
    print(f"Date        : {fields.get('date','')}")
    print(f"Age         : {fields.get('age','')}")
    print(f"Gender      : {fields.get('gender','')}")
    print(f"Doctor      : {fields.get('doctor','')}")
    print(f"Hospital    : {fields.get('hospital','')}")
    print(f"Amount      : {fields.get('amount','')}")
    meds = fields.get('medicines') or []
    if meds:
        print("Medicines:")
        for m in meds[:20]:
            print(f" - {m}")
    doses = fields.get('dosage') or []
    if doses:
        print("Dosage:")
        for d in doses[:20]:
            print(f" - {d}")
    if fields.get('notes'):
        print(f"Notes: {fields['notes']}")

    annotated = annotate(proc, boxes, fields)
    out_path = os.path.splitext(image_path)[0] + "_annotated.png"
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated preview saved to: {out_path}")

    try:
        cv2.imshow('Processed + Annotations', annotated)
        print("Press any key on the image window to exit…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        # Headless environments
        pass


if __name__ == '__main__':
    main()


