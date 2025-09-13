import whisper

model = whisper.load_model("medium")  # or "small", "base", etc.
result = model.transcribe("arabic_audio.wav", language="ar")
print(result["text"])
