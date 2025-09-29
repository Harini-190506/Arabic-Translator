from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from deep_translator import GoogleTranslator
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime, timezone
import logging
import os
import subprocess
import shutil
import speech_recognition as sr
from PIL import Image
import pytesseract
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'flac', 'aac', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

CORS(app, resources={
    r"/translate": {"origins": "*"},
    r"/transcribe": {"origins": "*"},
    r"/ocr": {"origins": "*"},
    r"/ocr_receipt": {"origins": "*"}
})

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check ffmpeg availability once at startup
FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None
if FFMPEG_AVAILABLE:
    logger.info("✅ ffmpeg detected on PATH")
else:
    logger.warning("⚠️ ffmpeg not found on PATH. Only WAV/FLAC inputs will be processed without conversion.")

# Check Tesseract availability once at startup
TESSERACT_AVAILABLE = shutil.which('tesseract') is not None
if TESSERACT_AVAILABLE:
    logger.info("✅ Tesseract OCR detected on PATH")
else:
    logger.warning("⚠️ Tesseract OCR not found on PATH. /ocr endpoint will fail until installed.")

# ---------------- LANGUAGES ----------------
LANGUAGES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Tamil": "ta",
    "Hindi": "hi",
    "Spanish": "es",
    "Arabic": "ar",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru"
}

# ---------------- HELPERS ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def preprocess_receipt(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")
    orig = image.copy()
    ratio = 1.0
    # Resize for consistent processing
    target_height = 900
    if image.shape[0] > target_height:
        ratio = image.shape[0] / target_height
        image = cv2.resize(image, (int(image.shape[1] / ratio), target_height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is not None:
        warped = four_point_transform(image, screenCnt.reshape(4, 2))
    else:
        warped = image
    # Convert to grayscale and enhance
    gray2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold improves text clarity
    thr = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    # Morphology to remove small noise
    kernel = np.ones((1, 1), np.uint8)
    clean = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    return clean

def image_to_base64(img_array):
    pil_img = Image.fromarray(img_array)
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def parse_medical_fields(text):
    import re
    data = {
        'patient_name': '',
        'date': '',
        'doctor': '',
        'hospital': '',
        'medicines': [],
        'notes': ''
    }
    # Simple heuristics for Arabic/English receipts
    # Date
    m = re.search(r"(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\d{4}[\-/]\d{1,2}[\-/]\d{1,2})", text)
    if m: data['date'] = m.group(0)
    # Patient keywords
    for kw in [r"اسم\s*المريض\s*[:：]?\s*(.+)", r"Patient\s*Name\s*[:：]?\s*(.+)"]:
        m = re.search(kw, text, re.IGNORECASE)
        if m:
            data['patient_name'] = m.group(1).split('\n')[0].strip()
            break
    # Doctor/Hospital
    m = re.search(r"(Dr\.?\s*[A-Za-z\u0621-\u064A]+[\sA-Za-z\u0621-\u064A]*)", text)
    if m: data['doctor'] = m.group(1)
    m = re.search(r"(Hospital|Clinic|مستشفى|عيادة)[^\n]*", text, re.IGNORECASE)
    if m: data['hospital'] = m.group(0)
    # Medicines: naive line-based parse for lines with mg/ml or Arabic units
    meds = []
    for line in text.splitlines():
        if re.search(r"(mg|ml|ملجم|ملغم|قرص|tablets?|caps?|ml)\b", line, re.IGNORECASE):
            meds.append(line.strip())
    data['medicines'] = meds
    # Notes
    data['notes'] = ''
    return data

def convert_to_wav(input_path, output_path):
    """Convert any audio file to WAV using ffmpeg"""
    try:
        if not FFMPEG_AVAILABLE:
            logger.error("FFmpeg unavailable: cannot convert %s to wav", input_path)
            return False
        result = subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', output_path
        ], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error("FFmpeg failed: output file missing or empty")
            return False
        logger.info(f"Conversion successful: {output_path}")
        return True
    except Exception as e:
        logger.error(f"convert_to_wav exception: {e}")
        return False

# ---------------- MONGO ----------------
MONGODB_AVAILABLE = False
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client['translation_db']
    MONGODB_AVAILABLE = True
    logger.info("✅ Connected to MongoDB")
except ConnectionFailure as e:
    logger.warning(f"⚠️ MongoDB not available: {e}")

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('index.html', languages=LANGUAGES)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    if not data or 'text' not in data or 'language' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    text = data['text'].strip()
    target_lang = data['language']
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if target_lang not in LANGUAGES.values():
        return jsonify({'error': 'Unsupported target language'}), 400
    
    try:
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
        translated_text = ""
        if is_arabic and target_lang == 'en':
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            translated_texts = []
            for sentence in sentences:
                try:
                    translator = GoogleTranslator(source='ar', target='en')
                    t = translator.translate(sentence)
                    translated_texts.append(t)
                except:
                    translated_texts.append(sentence)
            translated_text = '. '.join(translated_texts)
        else:
            translator = GoogleTranslator(source='auto', target=target_lang)
            translated_text = translator.translate(text)
        
        if MONGODB_AVAILABLE:
            try:
                db.translations.insert_one({
                    'original_text': text,
                    'translated_text': translated_text,
                    'source_lang': 'ar' if is_arabic else 'auto',
                    'target_lang': target_lang,
                    'timestamp': datetime.now(timezone.utc)
                })
            except Exception as e:
                logger.error(f"Mongo insert failed: {e}")
        
        return jsonify({'original_text': text, 'translated_text': translated_text, 'status': 'success'})
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'error': 'Translation failed', 'details': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Accept common field names or raw body
    audio_file = request.files.get('audio') or request.files.get('file')
    if audio_file is None and request.data:
        # Create a file-like object from raw bytes
        from io import BytesIO
        audio_file = type('RawFile', (), {
            'filename': 'upload.webm',
            'save': lambda self, dst: open(dst, 'wb').write(request.data)
        })()
    if audio_file is None:
        return jsonify({'error': 'No audio provided', 'status': 'error'}), 400
    
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400
    
    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file type', 'details': f"Got '{audio_file.filename}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", 'status': 'error'}), 400
    
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_audio')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, secure_filename(audio_file.filename))
    
    try:
        audio_file.save(temp_path)
        recognizer = sr.Recognizer()
        
        # Convert to WAV if needed
        if not temp_path.lower().endswith('.wav'):
            wav_path = os.path.splitext(temp_path)[0] + '.wav'
            if convert_to_wav(temp_path, wav_path):
                os.remove(temp_path)
                temp_path = wav_path
            else:
                return jsonify({
                    'error': 'Audio conversion failed',
                    'details': 'If uploading webm/mp3/ogg/m4a, ensure ffmpeg is installed and on PATH. Otherwise, upload WAV or FLAC.',
                    'status': 'error'
                }), 400
        
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language='ar-AR')
            except sr.UnknownValueError:
                try:
                    text = recognizer.recognize_google(audio_data, language='en-US')
                except sr.UnknownValueError:
                    return jsonify({'error': 'Could not understand audio', 'status':'error'}), 400
        
        return jsonify({'text': text, 'status': 'success'})
    
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return jsonify({'error': 'Failed to process audio', 'details': str(e), 'status':'error'}), 500
    
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Failed to delete temp file: {e}")

@app.route('/ocr', methods=['POST'])
def ocr_receipt():
    # Accept 'image' or common alternates
    image_file = request.files.get('image') or request.files.get('file') or request.files.get('photo')
    if image_file is None:
        return jsonify({'error': 'No image provided', 'status': 'error'}), 400

    if not TESSERACT_AVAILABLE:
        return jsonify({'error': 'Tesseract OCR is not installed', 'details': 'Install Tesseract and ensure it is on PATH. On Windows: winget install UB-Mannheim.Tesseract-OCR', 'status': 'error'}), 500

    if image_file.filename == '':
        return jsonify({'error': 'No selected image', 'status': 'error'}), 400

    if not allowed_image_file(image_file.filename):
        return jsonify({'error': 'Invalid image type', 'details': f"Got '{image_file.filename}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}", 'status': 'error'}), 400

    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_images')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, secure_filename(image_file.filename))

    try:
        image_file.save(temp_path)
        # Basic preprocessing: convert to grayscale
        img = Image.open(temp_path)
        img = img.convert('L')  # grayscale
        # Optional: further preprocessing could be added here
        lang = request.form.get('lang', 'eng')
        psm = request.form.get('psm')  # optional page segmentation mode
        config = ''
        if psm and str(psm).isdigit():
            config = f'--psm {psm}'
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        text = text.strip()
        return jsonify({'text': text, 'status': 'success'})
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return jsonify({'error': 'Failed to process image', 'details': str(e), 'status': 'error'}), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Failed to delete temp image: {e}")

@app.route('/ocr_receipt', methods=['POST'])
def ocr_receipt_advanced():
    if not TESSERACT_AVAILABLE:
        return jsonify({'error': 'Tesseract OCR is not installed', 'status': 'error'}), 500
    f = request.files.get('image') or request.files.get('file') or request.files.get('photo')
    if f is None:
        return jsonify({'error': 'No file provided', 'status': 'error'}), 400
    filename = secure_filename(f.filename or 'upload.png')
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    if ext == 'pdf':
        # Convert first page of PDF to image
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(f.read(), fmt='png')
            if not pages:
                return jsonify({'error': 'Empty PDF', 'status': 'error'}), 400
            buf = BytesIO()
            pages[0].save(buf, format='PNG')
            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_images')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, 'page1.png')
            cv2.imwrite(temp_path, img)
        except Exception as e:
            return jsonify({'error': 'PDF processing failed', 'details': str(e), 'status': 'error'}), 500
    else:
        if not allowed_image_file(filename):
            return jsonify({'error': 'Invalid image type', 'details': f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))} or PDF", 'status': 'error'}), 400
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_images')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        f.save(temp_path)
    try:
        clean = preprocess_receipt(temp_path)
        # Run OCR with Arabic + English
        data = pytesseract.image_to_data(clean, lang='ara+eng', config='--psm 6', output_type=pytesseract.Output.DICT)
        boxes = []
        full_text_lines = []
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            if not txt:
                continue
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            conf = float(data['conf'][i]) if data['conf'][i] not in ('-1', '') else -1
            boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': txt, 'conf': conf})
            full_text_lines.append(txt)
        full_text = '\n'.join(full_text_lines)
        fields = parse_medical_fields(full_text)
        preview_b64 = image_to_base64(clean)
        return jsonify({'status': 'success', 'preview_png_base64': preview_b64, 'boxes': boxes, 'fields': fields, 'raw_text': full_text})
    except Exception as e:
        logger.error(f"Advanced OCR failed: {e}")
        return jsonify({'error': 'Advanced OCR failed', 'details': str(e), 'status': 'error'}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

@app.route('/export_report', methods=['POST'])
def export_report():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'No JSON provided'}), 400
        fields = payload.get('fields') or {}
        translations = payload.get('translations') or {}
        export_format = (payload.get('format') or 'pdf').lower()

        if export_format == 'pdf':
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import mm
            buf = BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            width, height = A4
            y = height - 20*mm
            c.setFont('Helvetica-Bold', 16)
            c.drawString(20*mm, y, 'Medical Receipt Report')
            y -= 12*mm
            c.setFont('Helvetica', 11)
            def draw_kv(k, v):
                nonlocal y
                c.drawString(20*mm, y, f"{k}: {v or ''}")
                y -= 8*mm
            draw_kv('Patient Name', fields.get('patient_name'))
            draw_kv('Date', fields.get('date'))
            draw_kv('Doctor', fields.get('doctor'))
            draw_kv('Hospital', fields.get('hospital'))
            meds = '\n'.join(fields.get('medicines') or [])
            draw_kv('Medicines', '')
            for line in meds.split('\n'):
                c.drawString(26*mm, y, line)
                y -= 7*mm
            draw_kv('Notes', fields.get('notes'))
            y -= 6*mm
            c.setFont('Helvetica-Bold', 13)
            c.drawString(20*mm, y, 'Translation (EN)')
            y -= 9*mm
            c.setFont('Helvetica', 11)
            for k, v in translations.items():
                if isinstance(v, list):
                    c.drawString(20*mm, y, f"{k}:")
                    y -= 7*mm
                    for item in v:
                        c.drawString(26*mm, y, f"- {item}")
                        y -= 7*mm
                else:
                    c.drawString(20*mm, y, f"{k}: {v}")
                    y -= 7*mm
                if y < 20*mm:
                    c.showPage(); y = height - 20*mm; c.setFont('Helvetica', 11)
            c.showPage(); c.save()
            buf.seek(0)
            return app.response_class(buf.read(), mimetype='application/pdf', headers={
                'Content-Disposition': 'attachment; filename="receipt_report.pdf"'
            })

        elif export_format in ('xlsx', 'excel'):
            import openpyxl
            from openpyxl.utils import get_column_letter
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = 'Receipt'
            ws.append(['Field', 'Value'])
            ws.append(['Patient Name', fields.get('patient_name', '')])
            ws.append(['Date', fields.get('date', '')])
            ws.append(['Doctor', fields.get('doctor', '')])
            ws.append(['Hospital', fields.get('hospital', '')])
            ws.append(['Medicines', '\n'.join(fields.get('medicines') or [])])
            ws.append(['Notes', fields.get('notes', '')])
            ws2 = wb.create_sheet('Translation')
            ws2.append(['Field', 'English'])
            for k, v in translations.items():
                if isinstance(v, list):
                    ws2.append([k, '\n'.join(v)])
                else:
                    ws2.append([k, v])
            for wsx in (ws, ws2):
                for col in range(1, 3):
                    wsx.column_dimensions[get_column_letter(col)].width = 40
            buf = BytesIO()
            wb.save(buf)
            buf.seek(0)
            return app.response_class(buf.read(), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={
                'Content-Disposition': 'attachment; filename="receipt_report.xlsx"'
            })
        else:
            return jsonify({'error': 'Unsupported export format'}), 400
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({'error': 'Export failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
