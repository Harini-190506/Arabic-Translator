import subprocess
import shutil
import sys

def check_ffmpeg():
    print("Checking for ffmpeg installation...")
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        print(f"✅ FFmpeg found at: {ffmpeg_path}")
    else:
        print("❌ FFmpeg not found in system PATH")
    
    print("\nChecking for Python ffmpeg packages...")
    try:
        import ffmpeg
        print("✅ ffmpeg-python package is installed")
    except ImportError:
        print("❌ ffmpeg-python package is not installed")
    
    try:
        import ffmpeg_python
        print("✅ ffmpeg package is installed")
    except ImportError:
        print("❌ ffmpeg package is not installed")

if __name__ == "__main__":
    check_ffmpeg()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        print("\nAttempting to install ffmpeg-python...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        
        print("\nChecking again after installation:")
        check_ffmpeg()