import os
import sys
import logging
import wave
import contextlib
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from deep_translator import GoogleTranslator
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/transcribe": {"origins": "*"},
    r"/translate": {"origins": "*"}
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Language options
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
    "Russian": "ru",
}

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html', languages=LANGUAGES.keys())

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        text = data.get('text', '').strip()
        target_lang = data.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Translate the text
        translated = GoogleTranslator(source='auto', target=LANGUAGES.get(target_lang, 'en')).translate(text)
        return jsonify({'translated_text': translated})
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({'error': 'Translation failed. Please try again.'}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Check if the post request has the file part
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['audio']
    
    # If user does not select file, browser might submit an empty part
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only WAV files are allowed.'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Ensure the upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the uploaded file
        file.save(filepath)
        
        # Verify the file is not empty
        if os.path.getsize(filepath) == 0:
            raise Exception('Uploaded file is empty')
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        try:
            # Read the WAV file
            with sr.AudioFile(filepath) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                
                try:
                    # Recognize speech using Google Web Speech API
                    text = recognizer.recognize_google(audio_data)
                    return jsonify({'text': text})
                except sr.UnknownValueError:
                    return jsonify({'error': 'Could not understand the audio. The audio might be too quiet or unclear.'}), 400
                except sr.RequestError as e:
                    logger.error(f"Google Speech Recognition request failed: {str(e)}")
                    return jsonify({'error': 'Speech recognition service unavailable. Please try again later.'}), 503
                except Exception as e:
                    logger.error(f"Speech recognition error: {str(e)}")
                    return jsonify({'error': 'Error processing the audio file'}), 500
                
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return jsonify({'error': 'Failed to process the audio file. Please ensure it is a valid WAV file.'}), 400
            
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return jsonify({'error': str(e) or 'Failed to process the uploaded file'}), 500
        
    finally:
        # Clean up: remove the uploaded file after processing
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.error(f"Error cleaning up file {filepath}: {str(e)}")

if __name__ == '__main__':
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)
