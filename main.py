from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chatbot import chatbot
from cv import cv as cv_assistant
from quote import quote as quote_assistant, QuoteResponse
from news import get_spain_news, format_news_for_chat
from weather import get_weather, format_weather_for_chat
from spanish_newspapers import (
    get_combined_news, 
    format_newspapers_for_chat, 
    get_newspapers_by_source,
    get_elpais_news,
    get_elmundo_news
)
from voice import process_voice_message
from io import BytesIO
import base64

app = FastAPI()

class QuoteRequest(BaseModel):
    description: str
    interests: list[str]
    style: str
    language: str

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Hola Mundo"}

@app.get("/health")
async def health():
    return {"message": "saludable"}

@app.post("/chat")
async def chat(request: ChatRequest):
    response = chatbot(request.message)
    return {"response": response}


@app.post("/cv")
async def cv(request: ChatRequest):
    response = cv_assistant(request.message)
    return {"response": response}


@app.post("/quote")
async def quote(request: QuoteRequest) -> QuoteResponse:
    response = quote_assistant(request.description, request.interests)
    return response

@app.get("/news")
async def news(limit: int = 10):
    """
    Obtiene noticias recientes de España.
    
    Args:
        limit: Número máximo de noticias a retornar (por defecto 10, máximo recomendado 100)
    
    Returns:
        JSON con las noticias formateadas
    """
    # Limitar el máximo de noticias para evitar problemas
    limit = min(limit, 100)
    news_data = get_spain_news(limit=limit)
    return news_data

@app.get("/news/formatted")
async def news_formatted(limit: int = 10):
    """
    Obtiene noticias recientes de España formateadas para chat.
    
    Args:
        limit: Número máximo de noticias a retornar (por defecto 10)
    
    Returns:
        String formateado con las noticias listo para mostrar
    """
    limit = min(limit, 100)
    news_data = get_spain_news(limit=limit)
    formatted_text = format_news_for_chat(news_data)
    return {"response": formatted_text}

@app.get("/weather")
async def weather(city: str = "Madrid", country_code: str = "ES"):
    """
    Obtiene el clima actual de una ciudad.
    
    Args:
        city: Nombre de la ciudad (por defecto "Madrid")
        country_code: Código del país (por defecto "ES" para España)
    
    Returns:
        JSON con la información del clima
    """
    weather_data = get_weather(city=city, country_code=country_code)
    return weather_data

@app.get("/weather/formatted")
async def weather_formatted(city: str = "Madrid", country_code: str = "ES"):
    """
    Obtiene el clima actual de una ciudad formateado para chat.
    
    Args:
        city: Nombre de la ciudad (por defecto "Madrid")
        country_code: Código del país (por defecto "ES" para España)
    
    Returns:
        String formateado con la información del clima listo para mostrar
    """
    weather_data = get_weather(city=city, country_code=country_code)
    formatted_text = format_weather_for_chat(weather_data)
    return {"response": formatted_text}

@app.get("/newspapers")
async def newspapers(source: str = "ambos", limit: int = 5):
    """
    Obtiene noticias recientes directamente de El País y El Mundo.
    
    Args:
        source: Fuente de noticias - "ambos" (defecto), "elpais", o "elmundo"
        limit: Número de noticias por fuente (por defecto 5, si es "ambos" obtiene 5 de cada uno)
    
    Returns:
        JSON con las noticias actualizadas de los periódicos
    """
    if source.lower() == "ambos":
        news_data = get_combined_news(limit_per_source=limit)
    else:
        news_data = get_newspapers_by_source(source=source, limit=limit)
    
    return news_data

@app.get("/newspapers/formatted")
async def newspapers_formatted(source: str = "ambos", limit: int = 5):
    """
    Obtiene noticias de los periódicos formateadas para chat.
    
    Args:
        source: Fuente de noticias - "ambos" (defecto), "elpais", o "elmundo"
        limit: Número de noticias por fuente (por defecto 5)
    
    Returns:
        String formateado con las noticias listo para mostrar
    """
    if source.lower() == "ambos":
        news_data = get_combined_news(limit_per_source=limit)
    else:
        news_data = get_newspapers_by_source(source=source, limit=limit)
    
    formatted_text = format_newspapers_for_chat(news_data)
    return {"response": formatted_text}

@app.get("/newspapers/elpais")
async def elpais_only(limit: int = 10):
    """
    Obtiene noticias solo de El País.
    
    Args:
        limit: Número de noticias (por defecto 10)
    
    Returns:
        JSON con las noticias de El País
    """
    news = get_elpais_news(limit=limit)
    return {
        "total": len(news),
        "fuente": "El País",
        "news": news
    }

@app.get("/newspapers/elmundo")
async def elmundo_only(limit: int = 10):
    """
    Obtiene noticias solo de El Mundo.
    
    Args:
        limit: Número de noticias (por defecto 10)
    
    Returns:
        JSON con las noticias de El Mundo
    """
    news = get_elmundo_news(limit=limit)
    return {
        "total": len(news),
        "fuente": "El Mundo",
        "news": news
    }

@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    voice: str = "nova"
):
    """
    Procesa entrada de voz a través del chatbot y retorna respuesta en audio.
    
    Pipeline completo:
    1. Recibe archivo de audio del frontend
    2. Transcribe audio a texto (Speech-to-Text con Whisper)
    3. Procesa el texto a través del chatbot MenteViva
    4. Convierte la respuesta a audio (Text-to-Speech)
    5. Retorna el audio generado
    
    Args:
        audio: Archivo de audio (webm, mp3, wav, etc.)
        voice: Voz a usar para TTS - opciones: alloy, echo, fable, onyx, nova (default), shimmer
               'nova' es una voz cálida y amigable, ideal para usuarios mayores
    
    Returns:
        Audio MP3 con la respuesta del chatbot
    """
    try:
        # Validar el tipo de archivo
        if not audio.content_type or not audio.content_type.startswith('audio'):
            # Permitir también video/webm que es común para grabaciones de navegador
            if not audio.content_type.startswith('video'):
                raise HTTPException(
                    status_code=400, 
                    detail="El archivo debe ser de tipo audio o video/webm"
                )
        
        # Leer el archivo de audio
        audio_bytes = await audio.read()
        
        # Validar que el archivo no esté vacío
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="El archivo de audio está vacío")
        
        # Procesar a través del pipeline completo
        response_audio, transcribed_text, chatbot_response = process_voice_message(
            audio_bytes, 
            voice=voice
        )
        
        # Encode audio as base64 and return JSON with full response
        audio_base64 = base64.b64encode(response_audio).decode("utf-8")

        return {
            "text": transcribed_text,
            "response": chatbot_response,
            "audio": audio_base64,
            "audioType": "audio/mpeg"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando audio: {str(e)}")
