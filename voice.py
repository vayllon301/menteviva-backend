from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from io import BytesIO
from chatbot import chatbot_async

load_dotenv()

# Initialize async OpenAI client (non-blocking for FastAPI)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def transcribe_audio(audio_file: bytes) -> str:
    """
    Transcribe audio to text using OpenAI Whisper API (async).
    """
    try:
        audio_buffer = BytesIO(audio_file)
        audio_buffer.name = "audio.webm"

        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_buffer,
            language="es"
        )

        return transcript.text
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")


async def text_to_speech(text: str, voice: str = "nova") -> bytes:
    """
    Convert text to speech using OpenAI TTS API (async).
    Uses opus format for lower latency than MP3.
    """
    try:
        response = await client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="opus",
            speed=0.9
        )

        return response.content
    except Exception as e:
        raise Exception(f"Error generating speech: {str(e)}")


async def process_voice_message(
    audio_file: bytes,
    voice: str = "nova",
    history: list = None,
    user_profile: dict = None
) -> tuple:
    """
    Process voice message through the complete pipeline (async):
    1. Transcribe audio to text (STT)
    2. Process through chatbot (with history and profile)
    3. Convert response to speech (TTS)
    """
    # Step 1: Transcribe audio to text
    transcribed_text = await transcribe_audio(audio_file)

    # Step 2: Process through chatbot with context
    chatbot_response = await chatbot_async(
        transcribed_text,
        history=history,
        user_profile=user_profile
    )

    # Step 3: Convert response to speech
    response_audio = await text_to_speech(chatbot_response, voice=voice)

    return response_audio, transcribed_text, chatbot_response
