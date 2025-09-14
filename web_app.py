import os
import sys
import logging
import wave
import contextlib
import tempfile

# Use pocketsphinx for speech recognition instead of SpeechRecognition
try:
    import pocketsphinx
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print("Warning: pocketsphinx module could not be imported. Audio transcription will be disabled.")
    print(f"Error details: {e}")
    SPEECH_RECOGNITION_AVAILABLE = False

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from deep_translator import GoogleTranslator
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from datetime import datetime
from pymongo.errors import ConnectionFailure

app = Flask(__name__)

# Configure CORS to allow requests from the frontend
CORS(app, resources={
    r"/translate": {"origins": "*"},
    r"/transcribe": {"origins": "*"}
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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
    logger.error(f"❌ MongoDB connection failed: {e}")
    logger.warning("Application will run without database functionality")
    client = None
    db = None
    translations_collection = None

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

# Allowed file extensions - expanded to support all audio types
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'webm', 'aac', 'flac', 'wma', 'aiff', 'amr'}
ALLOWED_MIME_TYPES = {'audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/ogg', 'audio/webm', 
                     'audio/aac', 'audio/flac', 'audio/x-ms-wma', 'audio/aiff', 'audio/amr', 'audio/*'}


def allowed_file(filename, content_type=None):
    # Check file extension
    extension_valid = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # If content_type is provided, also check MIME type
    if content_type and extension_valid:
        # Check if the content type starts with any of our allowed audio types
        mime_valid = any(content_type.startswith(mime_type.split('/')[0]) for mime_type in ALLOWED_MIME_TYPES)
        return extension_valid and mime_valid
    
    return extension_valid

@app.route('/')
def home():
    return render_template('index.html', languages=LANGUAGES.keys())

@app.route('/test_ffmpeg')
def test_ffmpeg():
    """Test endpoint to verify FFmpeg functionality"""
    try:
        # Check if ffmpeg is installed
        import subprocess
        import shutil
        
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            # Check common installation locations
            potential_paths = [
                os.path.expanduser('~\\scoop\\shims\\ffmpeg.exe'),
                'C:\\ffmpeg\\bin\\ffmpeg.exe',
                'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
                'C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe'
            ]
            
            for path in potential_paths:
                if os.path.isfile(path):
                    ffmpeg_path = path
                    break
        
        if not ffmpeg_path:
            return jsonify({
                'status': 'error',
                'message': 'FFmpeg not found'
            }), 500
        
        # Create test input file
        test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        os.makedirs(test_folder, exist_ok=True)
        test_file = os.path.join(test_folder, 'test_audio.txt')
        if not os.path.exists(test_file):
            with open(test_file, 'w') as f:
                f.write('Test file for FFmpeg')
        
        # Create output file path
        output_file = os.path.join(test_folder, 'test_output.wav')
        
        # Use absolute path to ffmpeg executable
        if not ffmpeg_path:
            ffmpeg_path = os.path.expanduser('~\\scoop\\shims\\ffmpeg.exe')
        
        logger.info(f"Using FFmpeg at: {ffmpeg_path}")
        
        # Try to create a simple audio file
        cmd = [
            ffmpeg_path, '-y', '-f', 'lavfi', '-i', 'sine=frequency=1000:duration=5',
            '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', output_file
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
            return jsonify({
                'status': 'success',
                'message': f'FFmpeg test successful. Created test WAV file ({file_size} bytes)',
                'ffmpeg_path': ffmpeg_path
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'FFmpeg test failed: {result.stderr}',
                'ffmpeg_path': ffmpeg_path
            }), 500
    
    except Exception as e:
        logger.error(f"Error in test_ffmpeg: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Exception: {str(e)}'
        }), 500

@app.route('/translate', methods=['POST', 'OPTIONS'])
@cross_origin()
def translate():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'success'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text', '').strip()
        target_lang = data.get('language', 'English')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        logger.info(f"Translating text: {text} to {target_lang}")
        
        # Get the language code
        lang_code = LANGUAGES.get(target_lang, 'en')
        
        # Translate the text
        translated = GoogleTranslator(source='auto', target=lang_code).translate(text)
        
        # Save to MongoDB if connection is available
        saved_to_db = False
        if MONGODB_AVAILABLE and translations_collection is not None:
            try:
                translation_data = {
                    'original_text': text,
                    'translated_text': translated,
                    'timestamp': datetime.utcnow(),
                    'source_language': 'auto',
                    'target_language': lang_code
                }
                result = translations_collection.insert_one(translation_data)
                saved_to_db = True
                logger.info(f"✅ Saved translation with ID: {result.inserted_id}")
                logger.info(f"   Original: {text}")
                logger.info(f"   Translated: {translated}")
            except Exception as e:
                logger.error(f"❌ Failed to save to MongoDB: {e}")
                # Continue with translation even if DB save fails
        else:
            logger.info("MongoDB not available - translation will not be saved")
        
        return jsonify({
            'translated_text': translated,
            'saved_to_db': saved_to_db
        })
        
    except Exception as e:
        logger.error(f"❌ Translation error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Translation failed. Please try again.'}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Check if speech recognition is available
    if not SPEECH_RECOGNITION_AVAILABLE:
        logger.error("Speech recognition module is not available")
        return jsonify({
            'error': 'Speech recognition is not available on this server',
            'details': 'The pocketsphinx module could not be imported.'
        }), 503
        
    # Debug: Log request details
    logger.info(f"=== New Transcription Request ===")
    logger.info(f"Method: {request.method}")
    logger.info(f"Headers: {dict(request.headers)}")
    logger.info(f"Form data: {request.form}")
    logger.info(f"Files: {request.files}")
    
    # Check if the post request has the file part
    if 'audio' not in request.files:
        logger.error(f"No 'audio' in request.files. Available keys: {list(request.files.keys())}")
        return jsonify({
            'error': 'No file part in the request',
            'received_files': list(request.files.keys())
        }), 400
    
    file = request.files['audio']
    logger.info(f"Processing file: {file.filename}, Content-Type: {file.content_type}")
    
    # If user does not select file, browser might submit an empty part
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400
        
    # Check if the file type is allowed
    if not allowed_file(file.filename, file.content_type):
        logger.error(f"File type not allowed: {file.filename}, {file.content_type}")
        return jsonify({
            'error': 'File type not allowed',
            'details': f'Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    # Ensure the upload directory exists and is writable
    upload_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
    try:
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"Upload directory: {upload_dir} (exists: {os.path.exists(upload_dir)})")
        
        # Test if directory is writable
        test_file = os.path.join(upload_dir, '.test_write')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info("Upload directory is writable")
        except Exception as e:
            logger.error(f"Upload directory is not writable: {str(e)}")
            return jsonify({
                'error': 'Server configuration error',
                'details': f'Cannot write to upload directory: {str(e)}',
                'upload_dir': upload_dir,
                'cwd': os.getcwd()
            }), 500
        
    except Exception as e:
        logger.error(f"Error accessing upload directory {upload_dir}: {str(e)}")
        return jsonify({
            'error': 'Server configuration error',
            'details': str(e),
            'upload_dir': upload_dir,
            'cwd': os.getcwd()
        }), 500
    
    filepath = None
    try:
        # Create a secure filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        logger.info(f"Saving file to: {filepath}")
        
        # Save the file temporarily
        file.save(filepath)
        file_size = os.path.getsize(filepath)
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
        # Verify the file is not empty
        if file_size == 0:
            raise Exception('Uploaded file is empty')
        
        try:
            # Use pocketsphinx for speech recognition
            logger.info("Starting speech recognition with pocketsphinx...")
            
            # Create a temporary WAV file if needed (pocketsphinx works best with WAV)
            wav_filepath = filepath
            if not filepath.lower().endswith('.wav'):
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                        wav_filepath = temp_wav.name
                        logger.info(f"Converting audio to WAV format: {wav_filepath}")
                        
                        # Use ffmpeg to convert audio to WAV format
                        import subprocess
                        import shutil
                        
                        # Check if ffmpeg is installed
                        ffmpeg_path = shutil.which('ffmpeg')
                        
                        # If not found in PATH, check common installation locations
                        if not ffmpeg_path:
                            # Check Scoop installation path
                            potential_paths = [
                                os.path.expanduser('~\\scoop\\shims\\ffmpeg.exe'),
                                'C:\\ffmpeg\\bin\\ffmpeg.exe',
                                'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
                                'C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe'
                            ]
                            
                            for path in potential_paths:
                                if os.path.isfile(path):
                                    ffmpeg_path = path
                                    logger.info(f"Found FFmpeg at: {ffmpeg_path}")
                                    break
                        
                        if not ffmpeg_path:
                            logger.error("FFmpeg not found. Please install FFmpeg to enable audio conversion.")
                            return jsonify({
                                'error': 'FFmpeg not installed',
                                'details': 'FFmpeg is required to convert audio files. Please install FFmpeg on your system. For Windows, you can download it from https://ffmpeg.org/download.html or install it using a package manager like Chocolatey or Scoop.'
                            }), 500
                            
                        # Use absolute path to ffmpeg executable
                        if not ffmpeg_path:
                            ffmpeg_path = os.path.expanduser('~\\scoop\\shims\\ffmpeg.exe')
                            
                        logger.info(f"Using FFmpeg at: {ffmpeg_path}")
                        cmd = [
                            ffmpeg_path, '-y', '-i', filepath,
                            '-ac', '1',  # Convert to mono
                            '-ar', '16000',  # 16kHz sample rate
                            '-acodec', 'pcm_s16le',  # 16-bit PCM
                            wav_filepath
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.error(f"FFmpeg conversion error: {result.stderr}")
                            raise Exception(f"Failed to convert audio: {result.stderr}")
                        logger.info("Audio conversion successful")
                except Exception as e:
                    logger.error(f"Error converting audio: {str(e)}")
                    return jsonify({
                        'error': 'Failed to convert audio file',
                        'details': str(e)
                    }), 400
            
            # Try to detect language from audio
            # First try with Arabic model if available
            try:
                # Check if Arabic model exists
                arabic_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'arabic')
                use_arabic_model = os.path.exists(arabic_model_path)
                
                if use_arabic_model:
                    logger.info("Using Arabic speech model")
                    config = pocketsphinx.Config()
                    config.set_string("-hmm", arabic_model_path)
                    # Use appropriate Arabic language model and dictionary if available
                    # Note: You may need to install specific Arabic models
                    decoder = pocketsphinx.Decoder(config)
                else:
                    # Fallback to English model
                    logger.info("Arabic model not found, using English model")
                    config = pocketsphinx.Config()
                    config.set_string("-hmm", pocketsphinx.get_model_path() + "/en-us")
                    config.set_string("-lm", pocketsphinx.get_model_path() + "/en-us.lm.bin")
                    config.set_string("-dict", pocketsphinx.get_model_path() + "/cmudict-en-us.dict")
                    decoder = pocketsphinx.Decoder(config)
                
                # Process the audio file
                decoder.start_utt()
                with open(wav_filepath, 'rb') as f:
                    data = f.read()
                    decoder.process_raw(data, False, True)
                decoder.end_utt()
                
                # Get the transcription result
                hypothesis = decoder.hyp()
                if hypothesis:
                    text = hypothesis.hypstr
                    logger.info(f"Successfully transcribed audio: {text}")
                    
                    # Clean up temporary WAV file if it was created
                    if wav_filepath != filepath and os.path.exists(wav_filepath):
                        try:
                            os.remove(wav_filepath)
                            logger.info(f"Removed temporary WAV file: {wav_filepath}")
                        except Exception as e:
                            logger.error(f"Failed to remove temporary WAV file: {str(e)}")
                    
                    return jsonify({
                        'text': text,
                        'file_size': file_size,
                        'file_type': file.content_type
                    })
                else:
                    error_msg = 'Could not understand the audio. The audio might be too quiet or unclear.'
                    logger.error(error_msg)
                    return jsonify({'error': error_msg}), 400
                    
            except Exception as e:
                logger.error(f"Speech recognition error: {str(e)}")
                return jsonify({
                    'error': 'Failed to process the audio file',
                    'details': str(e)
                }), 400
                
        except Exception as e:
            error_msg = f'Error processing audio file: {str(e)}'
            logger.error(error_msg, exc_info=True)
            return jsonify({
                'error': 'Failed to process the audio file',
                'details': str(e),
                'file_type': file.content_type,
                'file_size': file_size
            }), 400
            
    except Exception as e:
        error_msg = f'File upload error: {str(e)}'
        logger.error(error_msg, exc_info=True)
        return jsonify({
            'error': 'Failed to process the uploaded file',
            'details': str(e),
            'filepath': filepath,
            'upload_dir': upload_dir
        }), 500
        
    finally:
        # Clean up: remove the uploaded file after processing
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Temporary file {filepath} removed")
            except Exception as e:
                logger.error(f"Error cleaning up file {filepath}: {str(e)}")

if __name__ == '__main__':
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)
