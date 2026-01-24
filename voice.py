from openai import OpenAI
from dotenv import load_dotenv
import os
from io import BytesIO
from chatbot import chatbot

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(audio_file: bytes) -> str:
    """
    Transcribe audio to text using OpenAI Whisper API.
    
    Args:
        audio_file: Audio file bytes
        
    Returns:
        Transcribed text
    """
    try:
        # Create a file-like object from bytes
        audio_buffer = BytesIO(audio_file)
        audio_buffer.name = "audio.webm"  # OpenAI needs a filename
        
        # Transcribe using Whisper
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_buffer,
            language="es"  # Spanish language for MenteViva
        )
        
        return transcript.text
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")


def text_to_speech(text: str, voice: str = "nova") -> bytes:
    """
    Convert text to speech using OpenAI TTS API.
    
    Args:
        text: Text to convert to speech
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
               'nova' is default - warm, friendly voice suitable for elderly users
        
    Returns:
        Audio file bytes (MP3 format)
    """
    try:
        # Generate speech using OpenAI TTS
        response = client.audio.speech.create(
            model="tts-1",  # Use tts-1-hd for higher quality if needed
            voice=voice,
            input=text,
            response_format="mp3",
            speed=0.9  # Slightly slower for better comprehension by elderly users
        )
        
        # Return the audio bytes
        return response.content
    except Exception as e:
        raise Exception(f"Error generating speech: {str(e)}")


def process_voice_message(audio_file: bytes, voice: str = "nova") -> bytes:
    """
    Process voice message through the complete pipeline:
    1. Transcribe audio to text (STT)
    2. Process through chatbot
    3. Convert response to speech (TTS)
    
    Args:
        audio_file: Input audio file bytes
        voice: Voice to use for TTS response
        
    Returns:
        Audio file bytes containing the chatbot's spoken response
    """
    # Step 1: Transcribe audio to text
    transcribed_text = transcribe_audio(audio_file)
    
    # Step 2: Process through chatbot
    chatbot_response = chatbot(transcribed_text)
    
    # Step 3: Convert response to speech
    response_audio = text_to_speech(chatbot_response, voice=voice)
    
    return response_audio, transcribed_text, chatbot_response
