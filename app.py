import os
import wave
import tempfile
import sys

# Use pocketsphinx for speech recognition instead of SpeechRecognition
try:
    import pocketsphinx
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print("Warning: pocketsphinx module could not be imported. Audio transcription will be disabled.")
    print(f"Error details: {e}")
    SPEECH_RECOGNITION_AVAILABLE = False

from tkinter import *
from tkinter import filedialog, messagebox
from deep_translator import GoogleTranslator
import threading
import subprocess
from pymongo import MongoClient
from datetime import datetime

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client['translation_db']
translations_collection = db['arabic_translator']

# Initialize the main window
root = Tk()
root.title("Language Translator ")
root.geometry("600x500")
root.config(bg="#f0f8ff")
root.resizable(True, True)

# Global variables
AUDIO_FILE_PATH = ""

# Create main frame
main_frame = Frame(root, bg="#f0f8ff")
main_frame.pack(expand=True, fill=BOTH, padx=20, pady=10)

# Input Section
input_frame = LabelFrame(main_frame, text=" Input ", font=("Helvetica", 11, "bold"), 
                        bg="#f0f8ff", fg="#2c3e50", bd=2)
input_frame.pack(fill=X, pady=(0, 10))

# Text input
input_text = Text(input_frame, height=8, width=60, font=("Helvetica", 10), 
                 wrap=WORD, bd=2, relief=SOLID, padx=10, pady=10)
input_text.pack(padx=10, pady=10, fill=BOTH, expand=True)

# Audio upload section
audio_frame = Frame(input_frame, bg="#f0f8ff")
audio_frame.pack(fill=X, pady=(0, 10))

# Divider
canvas = Canvas(audio_frame, height=20, bg="#f0f8ff", bd=0, highlightthickness=0)
canvas.create_line(10, 10, 200, 10, dash=(4, 2), fill="#95a5a6")
canvas.create_text(100, 10, text="OR", fill="#7f8c8d", font=("Helvetica", 9))
canvas.pack(fill=X, pady=5)

# Audio upload button
def browse_audio():
    global AUDIO_FILE_PATH
    filetypes = [
        ("All Audio Files", "*.wav *.mp3 *.ogg *.flac *.aac *.wma *.aiff *.amr *.m4a *.webm"),
        ("WAV files", "*.wav"),
        ("MP3 files", "*.mp3"),
        ("OGG files", "*.ogg"),
        ("FLAC files", "*.flac"),
        ("AAC files", "*.aac"),
        ("WMA files", "*.wma"),
        ("AIFF files", "*.aiff"),
        ("AMR files", "*.amr"),
        ("M4A files", "*.m4a"),
        ("WEBM files", "*.webm"),
        ("All files", "*.*")
    ]
    filepath = filedialog.askopenfilename(filetypes=filetypes)
    
    if filepath:
        try:
            # Check if the file is not a WAV file
            if not filepath.lower().endswith('.wav'):
                update_status("Converting audio to WAV format...")
                temp_wav = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
                success, msg = convert_audio_format(filepath, temp_wav)
                if not success:
                    raise ValueError(f"Failed to convert audio: {msg}")
                AUDIO_FILE_PATH = temp_wav
            else:
                AUDIO_FILE_PATH = filepath
                
            audio_label.config(text=os.path.basename(filepath), fg="#2c3e50")
            input_text.delete(1.0, END)
            update_status("Audio file loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio file: {str(e)}")
            update_status("Error loading audio file", True)
            audio_label.config(text="Error loading file", fg="#e74c3c")

upload_btn = Button(audio_frame, text="Upload Audio File", command=browse_audio,
                   bg="#3498db", fg="white", font=("Helvetica", 9, "bold"),
                   relief=FLAT, padx=10, pady=5)
upload_btn.pack(pady=5)

audio_label = Label(audio_frame, text="No file selected", fg="#7f8c8d", 
                   bg="#f0f8ff", font=("Helvetica", 8))
audio_label.pack()

# Add supported formats label
supported_formats = Label(audio_frame, text="Supports all audio formats (max 16MB)", 
                        fg="#7f8c8d", bg="#f0f8ff", font=("Helvetica", 7))
supported_formats.pack()

# Output Section
output_frame = LabelFrame(main_frame, text=" Translation ", font=("Helvetica", 11, "bold"),
                         bg="#f0f8ff", fg="#2c3e50", bd=2)
output_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

output_text = Text(output_frame, height=8, width=60, font=("Helvetica", 10), 
                  wrap=WORD, bd=2, relief=SOLID, padx=10, pady=10, fg="#27ae60")
output_text.pack(padx=10, pady=10, fill=BOTH, expand=True)

# Status bar
status_var = StringVar()
status_var.set("Ready")
status_bar = Label(root, textvariable=status_var, bd=1, relief=SUNKEN, anchor=W,
                  font=("Helvetica", 8), fg="#7f8c8d")
status_bar.pack(side=BOTTOM, fill=X)

# Controls Frame
controls_frame = Frame(main_frame, bg="#f0f8ff")
controls_frame.pack(fill=X, pady=10)

# Language selection
lang_frame = Frame(controls_frame, bg="#f0f8ff")
lang_frame.pack(side=LEFT, padx=(0, 20))

Label(lang_frame, text="Translate to:", font=("Helvetica", 10, "bold"), 
      bg="#f0f8ff").pack(anchor=W)

languages = {
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

lang_var = StringVar(root)
lang_var.set("English")  # Default

lang_menu = OptionMenu(lang_frame, lang_var, *languages.keys())
lang_menu.config(width=15, font=("Helvetica", 10))
lang_menu.pack(pady=5)

def update_status(message, is_error=False):
    status_var.set(message)
    status_bar.config(fg="#e74c3c" if is_error else "#7f8c8d")
    root.update_idletasks()

def is_valid_wav(filepath):
    try:
        with wave.open(filepath, 'rb') as wav_file:
            # Check basic WAV file properties
            if wav_file.getnchannels() not in (1, 2):
                return False, "Audio must be mono or stereo"
            if wav_file.getsampwidth() not in (1, 2, 4):
                return False, "Unsupported sample width"
            if wav_file.getframerate() < 8000 or wav_file.getframerate() > 48000:
                return False, "Sample rate must be between 8kHz and 48kHz"
            return True, ""
    except Exception as e:
        return False, f"Invalid WAV file: {str(e)}"

def convert_audio_format(input_file, output_file):
    try:
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
                    update_status(f"Found FFmpeg at: {ffmpeg_path}")
                    break
        
        if not ffmpeg_path:
            return False, "FFmpeg not installed. Please install FFmpeg to enable audio conversion. For Windows, you can download it from https://ffmpeg.org/download.html or install it using a package manager like Chocolatey or Scoop."
        
        try:
            # Try to import ffmpeg-python
            import ffmpeg
            
            # Use ffmpeg-python for more robust audio conversion
            try:
                # First try with ffmpeg-python for better error handling
                (ffmpeg
                    .input(input_file)
                    .output(output_file, 
                            acodec='pcm_s16le',  # 16-bit PCM
                            ac=1,               # mono
                            ar=16000)           # 16kHz sample rate
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                return True, ""
            except ffmpeg.Error as e:
                # Fall back to subprocess if ffmpeg-python fails
                update_status("Falling back to direct ffmpeg command...")
                # Use absolute path to ffmpeg executable
                if not ffmpeg_path:
                    ffmpeg_path = os.path.expanduser('~\\scoop\\shims\\ffmpeg.exe')
                    
                print(f"Using FFmpeg at: {ffmpeg_path}")
                cmd = [
                    ffmpeg_path, '-y', '-i', input_file,
                    '-ac', '1',  # Convert to mono
                    '-ar', '16000',  # 16kHz sample rate
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    output_file
                ]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                return True, ""
        except ImportError:
            # If ffmpeg-python is not installed, use subprocess directly
            update_status("Using direct ffmpeg command...")
            cmd = [
                ffmpeg_path, '-y', '-i', input_file,
                '-ac', '1',  # Convert to mono
                '-ar', '16000',  # 16kHz sample rate
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                output_file
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True, ""
    except Exception as e:
        return False, f"Failed to convert audio: {str(e)}"

def transcribe_audio():
    try:
        # Check if speech recognition is available
        if not SPEECH_RECOGNITION_AVAILABLE:
            messagebox.showerror("Error", "Speech recognition is not available. Please install pocketsphinx.")
            return None
            
        if not AUDIO_FILE_PATH:
            raise ValueError("No audio file selected")
            
        update_status("Checking audio file...")
        
        # Validate WAV file
        is_valid, error_msg = is_valid_wav(AUDIO_FILE_PATH)
        if not is_valid:
            # Try to convert the file
            temp_file = os.path.join(os.path.dirname(AUDIO_FILE_PATH), "temp_converted.wav")
            success, convert_msg = convert_audio_format(AUDIO_FILE_PATH, temp_file)
            if not success:
                raise ValueError(f"{error_msg}. {convert_msg}")
            
            # Use the converted file
            audio_file = temp_file
            update_status("Converted audio file for better compatibility...")
        else:
            audio_file = AUDIO_FILE_PATH
            
        update_status("Transcribing audio...")
        
        # Check if Arabic model exists
        arabic_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'arabic')
        use_arabic_model = os.path.exists(arabic_model_path)
        
        # Try with Arabic model first if available
        text = None
        if use_arabic_model:
            try:
                update_status("Attempting transcription with Arabic model...")
                # Create a decoder with Arabic configuration
                config = pocketsphinx.Config()
                config.set_string("-hmm", arabic_model_path)
                # Use appropriate Arabic language model and dictionary if available
                decoder = pocketsphinx.Decoder(config)
                
                # Process the audio file
                decoder.start_utt()
                with open(audio_file, 'rb') as f:
                    data = f.read()
                    decoder.process_raw(data, False, True)
                decoder.end_utt()
                
                # Get the transcription result
                hypothesis = decoder.hyp()
                if hypothesis:
                    text = hypothesis.hypstr
                    update_status("Successfully transcribed with Arabic model")
            except Exception as e:
                update_status(f"Arabic model transcription failed: {str(e)}")
                # Continue to try with English model
        
        # If Arabic transcription failed or wasn't available, try with English model
        if text is None:
            update_status("Attempting transcription with English model...")
            # Create a decoder with default configuration
            config = pocketsphinx.Config()
            # Use English language model
            config.set_string("-hmm", pocketsphinx.get_model_path() + "/en-us")
            config.set_string("-lm", pocketsphinx.get_model_path() + "/en-us.lm.bin")
            config.set_string("-dict", pocketsphinx.get_model_path() + "/cmudict-en-us.dict")
            decoder = pocketsphinx.Decoder(config)
            
            # Process the audio file
            decoder.start_utt()
            with open(audio_file, 'rb') as f:
                data = f.read()
                decoder.process_raw(data, False, True)
            decoder.end_utt()
            
            # Get the transcription result
            hypothesis = decoder.hyp()
            if hypothesis:
                text = hypothesis.hypstr
                update_status("Successfully transcribed with English model")
            else:
                raise ValueError("Could not understand audio with any available model")
        
        # Display the transcribed text
        if text:
            input_text.delete(1.0, END)
            input_text.insert(END, text)
            
            # Clean up temporary file if it exists
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
                
            return text
        else:
            raise ValueError("Transcription failed with all available models")
        
    except Exception as e:
        update_status(f"Error in transcription: {str(e)}", True)
        messagebox.showerror("Transcription Error", f"Failed to transcribe audio: {str(e)}")
        return None

def translate_text():
    text = input_text.get(1.0, END).strip()
    
    if not text and not AUDIO_FILE_PATH:
        messagebox.showwarning("Warning", "Please enter text or upload an audio file")
        return
    
    update_status("Translating...")
    
    try:
        if AUDIO_FILE_PATH:
            text = transcribe_audio()
            if not text:
                return
        
        # Get the selected language code
        selected_lang = lang_var.get()
        target_lang_code = languages.get(selected_lang, 'en')
        
        # Get the translation
        translated = GoogleTranslator(source='auto', target=target_lang_code).translate(text)
        
        # Save to MongoDB
        translation_data = {
            'original_text': text,
            'translated_text': translated,
            'timestamp': datetime.now(),
            'source_language': 'auto',
            'target_language': target_lang_code
        }
        
        # Insert the translation into MongoDB
        result = translations_collection.insert_one(translation_data)
        
        # Update the UI
        output_text.config(state=NORMAL)
        output_text.delete(1.0, END)
        output_text.insert(END, translated)
        output_text.config(state=DISABLED)
        
        update_status(f"Translation completed and saved (ID: {result.inserted_id})")
        
    except Exception as e:
        messagebox.showerror("Error", f"Translation failed: {str(e)}")
        update_status("Translation failed", True)

# Buttons
button_frame = Frame(controls_frame, bg="#f0f8ff")
button_frame.pack(side=RIGHT)

# Clear button
def clear_all():
    global AUDIO_FILE_PATH
    input_text.delete(1.0, END)
    output_text.delete(1.0, END)
    audio_label.config(text="No file selected")
    AUDIO_FILE_PATH = ""
    update_status("Cleared all fields")

clear_btn = Button(button_frame, text="Clear All", command=clear_all,
                  font=("Helvetica", 10), bg="#95a5a6", fg="white",
                  relief=FLAT, padx=15, pady=5)
clear_btn.pack(side=RIGHT, padx=5)

# Translate button
translate_btn = Button(button_frame, text="Translate", command=translate_text,
                      font=("Helvetica", 10, "bold"), bg="#2ecc71", fg="white",
                      relief=FLAT, padx=20, pady=5)
translate_btn.pack(side=RIGHT)

# Configure grid weights
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

# Set minimum window size
root.update()
root.minsize(600, 500)

# Center the window on screen
window_width = root.winfo_width()
window_height = root.winfo_height()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f'+{x}+{y}')

# Run the GUI loop
root.mainloop()
