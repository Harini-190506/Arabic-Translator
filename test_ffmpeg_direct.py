import os
import subprocess
import shutil
import tempfile

def main():
    print("Testing direct FFmpeg usage")
    
    # Check if ffmpeg is in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        print(f"FFmpeg found in PATH: {ffmpeg_path}")
    else:
        print("FFmpeg not found in PATH, checking common locations...")
        
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
                print(f"Found FFmpeg at: {ffmpeg_path}")
                break
    
    if not ffmpeg_path:
        print("ERROR: FFmpeg not found!")
        return
    
    # Create a test file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
        temp.write(b"Test file for FFmpeg")
        test_file = temp.name
    
    # Create output file path
    output_file = test_file + ".wav"
    
    try:
        # Try to run FFmpeg directly
        print(f"\nRunning FFmpeg command: {ffmpeg_path} -version")
        result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
        print(f"Return code: {result.returncode}")
        print(f"Output: {result.stdout[:200]}..." if len(result.stdout) > 200 else f"Output: {result.stdout}")
        
        if result.stderr:
            print(f"Error: {result.stderr}")
        
        # Try to create a simple audio file
        print(f"\nTrying to create a test WAV file with FFmpeg")
        cmd = [
            ffmpeg_path, '-y', '-f', 'lavfi', '-i', 'sine=frequency=1000:duration=5',
            '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', output_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Success! Created test WAV file at: {output_file}")
            print(f"File size: {os.path.getsize(output_file)} bytes")
        else:
            print(f"Failed to create test WAV file. Error: {result.stderr}")
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
    
    finally:
        # Clean up
        try:
            os.unlink(test_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
        except:
            pass

if __name__ == "__main__":
    main()