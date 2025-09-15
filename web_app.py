from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from deep_translator import GoogleTranslator
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime, timezone
import logging
import os
import tempfile
import subprocess
import speech_recognition as sr
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'flac', 'aac'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Configure CORS
CORS(app, resources={
    r"/translate": {"origins": "*"},
    r"/transcribe": {"origins": "*"}
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Language configuration
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path, output_path):
    """Convert any audio file to WAV format using ffmpeg"""
    try:
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("FFmpeg is not installed or not in PATH")
            return False
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert the file
        result = subprocess.run([
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', input_path,  # Input file
            '-ar', '16000',    # Sample rate
            '-ac', '1',        # Mono channel
            '-c:a', 'pcm_s16le',  # 16-bit PCM
            output_path        # Output file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            return False
            
        # Verify the output file was created
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error("FFmpeg conversion failed: Output file not created or is empty")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error in convert_to_wav: {str(e)}")
        return False

# MongoDB Configuration
MONGODB_AVAILABLE = False
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.admin.command('ping')  # Test the connection
    db = client['translation_db']
    translations_collection = db['arabic_translator']
    MONGODB_AVAILABLE = True
    logger.info("✅ Successfully connected to MongoDB")
except ConnectionFailure as e:
    logger.warning(f"⚠️ Could not connect to MongoDB: {e}")
    logger.warning("Translation history will not be saved")

@app.route('/')
def home():
    """Render the main page with available languages."""
    return render_template('index.html', languages=LANGUAGES)

@app.route('/translate', methods=['POST'])
def translate():
    """Handle text translation requests with improved accuracy for Arabic."""
    data = request.get_json()
    if not data or 'text' not in data or 'language' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    text = data['text'].strip()
    target_lang = data['language']
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Validate language
    if target_lang not in LANGUAGES.values():
        return jsonify({'error': 'Unsupported target language'}), 400
    
    try:
        # Detect if the text is Arabic
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
        
        # For Arabic to English translation, use specific handling
        if is_arabic and target_lang == 'en':
            # Split text into sentences for better translation
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            translated_sentences = []
            
            for sentence in sentences:
                try:
                    # First try with Google Translator
                    translator = GoogleTranslator(source='ar', target='en')
                    translated = translator.translate(sentence)
                    
                    # Common fixes for known mistranslations
                    translation_fixes = {
                        'removing traditional poetry': 'traditional hair removal',
                        'depends on the thermal technique where it is stopped with the soul': 'relies on thermal technology',
                        'remote hair': 'hair removal',
                        'Nono': 'Nono',  # Preserve brand name
                        'Orbit Showtime': 'Orbit Showtime'  # Preserve brand name
                    }
                    
                    # Apply fixes if needed
                    for wrong, correct in translation_fixes.items():
                        translated = translated.replace(wrong, correct)
                        
                    translated_sentences.append(translated)
                except Exception as e:
                    logger.error(f"Error translating sentence: {sentence}. Error: {str(e)}")
                    translated_sentences.append(sentence)  # Keep original if translation fails
            
            translated_text = '. '.join(translated_sentences)
        else:
            # For other language pairs, use default translation
            translator = GoogleTranslator(source='auto', target=target_lang)
            translated_text = translator.translate(text)
        
        # Save to MongoDB if available
        if MONGODB_AVAILABLE:
            try:
                translation_record = {
                    'original_text': text,
                    'translated_text': translated_text,
                    'source_lang': 'ar' if is_arabic else 'auto',
                    'target_lang': target_lang,
                    'timestamp': datetime.now(timezone.utc)
                }
                db.translations.insert_one(translation_record)
            except Exception as e:
                logger.error(f"Error saving to MongoDB: {str(e)}")
        
        return jsonify({
            'original_text': text,
            'translated_text': translated_text,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({
            'error': 'Failed to translate text',
            'details': str(e)
        }), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Handle audio file upload and transcription with improved error handling."""
    logger.info("Received transcription request")
    
    # Check if the post request has the file part
    if 'audio' not in request.files:
        logger.error("No audio file part in the request")
        return jsonify({'error': 'No audio file provided', 'status': 'error'}), 400
    
    audio_file = request.files['audio']
    
    # If user does not select file, browser might submit an empty part without filename
    if audio_file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400
    
    # Check file extension
    allowed_extensions = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac'}
    if '.' not in audio_file.filename or \
       audio_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        logger.error(f"File type not allowed: {audio_file.filename}")
        return jsonify({
            'error': 'File type not allowed. Please upload an audio file (MP3, WAV, OGG, M4A, FLAC, AAC)',
            'status': 'error'
        }), 400
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_audio')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, secure_filename(audio_file.filename))
    
    try:
        # Save the uploaded file
        audio_file.save(temp_path)
        logger.info(f"Saved uploaded file to: {temp_path}")
        
        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        
        try:
            # If the file is not WAV, try to convert it
            if not temp_path.lower().endswith('.wav'):
                wav_path = os.path.splitext(temp_path)[0] + '.wav'
                logger.info(f"Converting {temp_path} to WAV format")
                if convert_to_wav(temp_path, wav_path):
                    # If conversion successful, use the WAV file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    temp_path = wav_path
                else:
                    logger.warning("FFmpeg conversion failed, trying direct processing")
                    # If conversion fails, try processing the original file
                    if not temp_path.lower().endswith(('.wav', '.flac')):
                        return jsonify({
                            'error': 'Failed to convert audio file. Please upload a WAV or FLAC file or install FFmpeg.',
                            'status': 'error'
                        }), 400
            
            # Process the audio file
            with sr.AudioFile(temp_path) as source:
                # Listen for the data (load audio to memory)
                audio_data = recognizer.record(source)
                logger.info("Attempting to transcribe audio...")
                
                # First try with Arabic language
                try:
                    text = recognizer.recognize_google(audio_data, language='ar-AR')
                    logger.info("Successfully transcribed audio with Arabic language model")
                except sr.UnknownValueError:
                    # If Arabic fails, try with English
                    try:
                        text = recognizer.recognize_google(audio_data, language='en-US')
                        logger.info("Successfully transcribed audio with English language model")
                    except sr.UnknownValueError:
                        logger.error("Could not understand the audio")
                        return jsonify({
                            'error': 'Could not understand the audio. Please ensure the audio is clear and try again.',
                            'status': 'error'
                        }), 400
                
                logger.info(f"Successfully transcribed audio: {text[:100]}...")
                
                return jsonify({
                    'text': text,
                    'status': 'success'
                })
                
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return jsonify({
                'error': 'Speech recognition service error. Please try again later.',
                'details': str(e),
                'status': 'error'
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        return jsonify({
            'error': 'Failed to process audio file. Please try again.',
            'details': str(e),
            'status': 'error'
        }), 500
        
    finally:
        # Clean up temporary files
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_path}: {str(e)}")

if __name__ == '__main__':
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)
