import os
from gtts import gTTS
import io

def transcribe_audio(client, audio_bytes):
    # Groq API for STT using whisper-large-v3
    # client is a groq.Groq instance
    # create a file-like object
    audio_file = ("audio.wav", audio_bytes, "audio/wav")
    try:
        completion = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
        )
        return completion.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def get_chat_response(client, messages):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def text_to_audio(text, lang='en'):
    # Detect language basic heuristic: if it contains Devanagari chars, use 'hi', else 'en'
    # Actually wait, let's allow explicit language selection or heuristic
    def has_hindi(text):
        return any('\u0900' <= char <= '\u097F' for char in text)
    
    detected_lang = 'hi' if has_hindi(text) else 'en'
    
    try:
        tts = gTTS(text=text, lang=detected_lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        return None
