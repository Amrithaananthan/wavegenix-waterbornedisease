# D:\sihdeeo\blockchain_messaging\language_support.py
import os
import time
from gtts import gTTS
import pygame
from .config import AUDIO_DIR

class LanguageSupport:
    def __init__(self):
        self.audio_dir = AUDIO_DIR
        os.makedirs(self.audio_dir, exist_ok=True)
        pygame.mixer.init()
    
    def text_to_speech(self, message, lang_code="en", play_audio=True):
        """Convert text to speech using gTTS (supports Hindi)"""
        try:
            # Create filename with timestamp
            timestamp = str(int(time.time()))
            filename = f"alert_{lang_code}_{timestamp}.mp3"
            filepath = os.path.join(self.audio_dir, filename)
            
            # Generate speech
            tts = gTTS(text=message, lang=lang_code, slow=False)
            tts.save(filepath)
            
            # Play audio if requested
            if play_audio:
                self._play_audio(filepath)
            
            return filepath
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return None
    
    def _play_audio(self, filepath):
        """Play audio file using pygame"""
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            # Wait for audio to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Audio playback error: {e}")

# Singleton instance
language_support = LanguageSupport()