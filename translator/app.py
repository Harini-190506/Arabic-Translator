# translator_gui.py
from tkinter import *
from deep_translator import GoogleTranslator

# Initialize the main window
root = Tk()
root.title("Language Translator 🌍")
root.geometry("500x400")
root.config(bg="#e0f7fa")

# Labels
Label(root, text="Enter Text:", font=("Helvetica", 12, "bold"), bg="#e0f7fa").pack(pady=10)
input_text = Text(root, height=5, width=50)
input_text.pack()

Label(root, text="Translated Text:", font=("Helvetica", 12, "bold"), bg="#e0f7fa").pack(pady=10)
output_text = Text(root, height=5, width=50, fg="green")
output_text.pack()

# Dropdown for language selection
Label(root, text="Translate to:", font=("Helvetica", 12), bg="#e0f7fa").pack(pady=5)

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

OptionMenu(root, lang_var, *languages.keys()).pack()

# Translate function
def translate_text():
    try:
        text = input_text.get("1.0", END).strip()
        target_lang = languages[lang_var.get()]
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        output_text.delete("1.0", END)
        output_text.insert(END, translated)
    except Exception as e:
        output_text.delete("1.0", END)
        output_text.insert(END, f"Error: {str(e)}")

# Button
Button(root, text="Translate", command=translate_text, font=("Helvetica", 12, "bold"), bg="#00796b", fg="white").pack(pady=20)

# Run the GUI loop
root.mainloop()
