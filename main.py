import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chatbot import chatbot, chatbot_async, chatbot_stream
from news import get_spain_news, format_news_for_chat
from weather import get_weather, format_weather_for_chat
from spanish_newspapers import (
    get_combined_news,
    format_newspapers_for_chat,
    get_newspapers_by_source,
    get_news_by_source,
    RSS_SOURCES,
)
from voice import process_voice_message, transcribe_audio, text_to_speech
from alert import send_whatsapp_alert
from memory_service import run_memory_pipeline
from instagram import validate_instagram_username, fetch_instagram_profile
from reminders import (
    list_active_reminders,
    create_reminder,
    update_reminder,
    get_unread_notifications,
    mark_notification_read,
)
from reminder_scheduler import scheduler_loop, run_tick
from io import BytesIO
import base64
import json


@asynccontextmanager
async def lifespan(app):
    # The in-process scheduler is unreliable on Azure App Service (the worker
    # gets unloaded on inactivity / restarted, killing the background task).
    # In production an external cron should hit POST /scheduler/tick instead.
    # Set RUN_INPROCESS_SCHEDULER=1 in local .env to keep the in-process loop
    # for development.
    task = None
    if os.getenv("RUN_INPROCESS_SCHEDULER") == "1":
        task = asyncio.create_task(scheduler_loop())
    yield
    if task is not None:
        task.cancel()

app = FastAPI(lifespan=lifespan)

SCHEDULER_SECRET = os.getenv("SCHEDULER_SECRET", "")

class AlertRequest(BaseModel):
    to: Optional[str] = None

class UserProfile(BaseModel):
    name: str
    number: Optional[str] = None
    description: Optional[str] = None
    interests: Optional[str] = None
    city: Optional[str] = None

class TutorProfile(BaseModel):
    name: str
    number: Optional[str] = None
    description: Optional[str] = None
    instagram: Optional[str] = None
    facebook: Optional[str] = None
    relationship: Optional[str] = None
    factors: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    user_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None
    tutor_profile: Optional[TutorProfile] = None
    user_memory: Optional[dict] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class InstagramLinkRequest(BaseModel):
    username: str

class MemorySummarizeRequest(BaseModel):
    user_id: str
    messages: List[dict]


class ReminderCreateRequest(BaseModel):
    user_id: str
    message: str
    remind_at: str
    recurrence: Optional[str] = None
    created_by: str = "tutor"


class ReminderSnoozeRequest(BaseModel):
    minutes: int = 10

@app.get("/")
async def root():
    return {"message": "Hola Mundo"}

@app.get("/health")
async def health():
    return {"message": "saludable"}

@app.post("/chat")
async def chat(request: ChatRequest):
    profile = request.user_profile.model_dump() if request.user_profile else {}
    if request.user_id:
        profile["id"] = request.user_id
    tutor = request.tutor_profile.model_dump() if request.tutor_profile else None
    user_location = {}
    if request.latitude is not None and request.longitude is not None:
        user_location = {"latitude": request.latitude, "longitude": request.longitude}
    response = await chatbot_async(request.message, history=request.history, user_profile=profile or None, tutor_profile=tutor, user_memory=request.user_memory, user_location=user_location)
    return {"response": response}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    profile = request.user_profile.model_dump() if request.user_profile else {}
    if request.user_id:
        profile["id"] = request.user_id
    tutor = request.tutor_profile.model_dump() if request.tutor_profile else None
    user_location = {}
    if request.latitude is not None and request.longitude is not None:
        user_location = {"latitude": request.latitude, "longitude": request.longitude}

    async def event_generator():
        try:
            async for token in chatbot_stream(
                request.message,
                history=request.history,
                user_profile=profile,
                tutor_profile=tutor,
                user_memory=request.user_memory,
                user_location=user_location,
            ):
                # SSE format: each event is "data: <content>\n\n"
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/memory/summarize", status_code=202)
async def summarize_memory(request: MemorySummarizeRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_memory_pipeline, request.user_id, request.messages)
    return {"status": "accepted"}


@app.get("/reminders/{user_id}")
async def get_reminders(user_id: str):
    """List active reminders for a user."""
    reminders_list = await list_active_reminders(user_id)
    return {"reminders": reminders_list}


@app.post("/reminders")
async def post_reminder(request: ReminderCreateRequest):
    """Create a reminder (used by tutor dashboard)."""
    result = await create_reminder(
        user_id=request.user_id,
        message=request.message,
        remind_at=request.remind_at,
        recurrence=request.recurrence,
        created_by=request.created_by,
    )
    return {"reminder": result}


@app.patch("/reminders/{reminder_id}/snooze")
async def snooze_reminder(reminder_id: str, request: ReminderSnoozeRequest):
    """Snooze a reminder by N minutes."""
    from datetime import datetime, timedelta, timezone

    new_time = datetime.now(timezone.utc) + timedelta(minutes=request.minutes)
    await update_reminder(reminder_id, {
        "status": "active",
        "remind_at": new_time.isoformat(),
    })
    return {"snoozed_until": new_time.isoformat()}


@app.patch("/reminders/{reminder_id}/dismiss")
async def dismiss_reminder(reminder_id: str):
    """Dismiss (complete) a reminder."""
    await update_reminder(reminder_id, {"status": "completed"})
    return {"status": "completed"}


@app.get("/notifications/{user_id}")
async def get_notifications(user_id: str):
    """Get unread notifications for the in-app popup."""
    notifications = await get_unread_notifications(user_id)
    return {"notifications": notifications}


@app.patch("/notifications/{notification_id}/read")
async def read_notification(notification_id: str):
    """Mark a notification as read."""
    await mark_notification_read(notification_id)
    return {"status": "read"}


@app.post("/scheduler/tick")
async def scheduler_tick(authorization: Optional[str] = Header(None)):
    """Run one polling iteration of the reminder scheduler.

    Intended to be called by an external cron (cron-job.org, GitHub Actions,
    Azure Logic App, etc.) every minute. Requires SCHEDULER_SECRET env var
    and an `Authorization: Bearer <secret>` header.
    """
    if not SCHEDULER_SECRET:
        raise HTTPException(status_code=503, detail="Scheduler not configured")
    if authorization != f"Bearer {SCHEDULER_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    processed = await run_tick()
    return {"processed": processed}


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
async def newspapers(source: str = "todos", limit: int = 3):
    """
    Obtiene noticias de periódicos españoles.

    Args:
        source: "todos" (defecto) o clave del periódico (elpais, elmundo, larazon,
                elperiodico, lavanguardia, abc, elespanol, elconfidencial, eldiario, mundodeportivo)
        limit: Noticias por fuente (por defecto 3)
    """
    if source.lower() == "todos":
        news_data = get_combined_news(limit_per_source=limit)
    else:
        news_data = get_newspapers_by_source(source=source, limit=limit)
    return news_data

@app.get("/newspapers/formatted")
async def newspapers_formatted(source: str = "todos", limit: int = 3):
    """
    Obtiene noticias de los periódicos formateadas para chat.
    """
    if source.lower() == "todos":
        news_data = get_combined_news(limit_per_source=limit)
    else:
        news_data = get_newspapers_by_source(source=source, limit=limit)
    formatted_text = format_newspapers_for_chat(news_data)
    return {"response": formatted_text}

@app.get("/newspapers/sources")
async def newspapers_sources():
    """Lista los periódicos disponibles."""
    return {source_key: info["name"] for source_key, info in RSS_SOURCES.items()}

@app.get("/newspapers/{source_key}")
async def newspaper_by_source(source_key: str, limit: int = 10):
    """
    Obtiene noticias de un periódico específico por su clave.
    """
    news_data = get_newspapers_by_source(source=source_key, limit=limit)
    return news_data

@app.post("/alert")
async def alert(request: AlertRequest):
    """
    Envía una alerta por WhatsApp usando Twilio.

    Args:
        request: JSON con 'message' (obligatorio) y 'to' (opcional, formato 'whatsapp:+34XXXXXXXXX')

    Returns:
        JSON con el resultado del envío
    """
    result = send_whatsapp_alert(to=request.to)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/tutor/instagram/link")
async def link_instagram(request: InstagramLinkRequest):
    """
    Valida y vincula una cuenta de Instagram al perfil del tutor.

    Acepta un nombre de usuario, @usuario, o URL completa de Instagram.
    Valida el formato y opcionalmente verifica que el perfil exista.

    Returns:
        JSON con el username validado, URL del perfil, y estado de verificación
    """
    validation = validate_instagram_username(request.username)

    if validation.get("error"):
        raise HTTPException(status_code=400, detail=validation["error"])

    username = validation["username"]

    # Try to verify the profile exists (best-effort, non-blocking)
    profile_check = fetch_instagram_profile(username)

    return {
        "linked": True,
        "username": username,
        "profile_url": validation["profile_url"],
        "verified": profile_check["exists"] if profile_check else None,
    }


@app.post("/tutor/instagram/unlink")
async def unlink_instagram():
    """
    Desvincula la cuenta de Instagram del perfil del tutor.

    Returns:
        JSON confirmando la desvinculación
    """
    return {
        "linked": False,
        "username": None,
        "profile_url": None,
    }


@app.post("/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
):
    """
    Transcribe audio a texto (solo STT). Endpoint rápido (~1-2s).
    """
    try:
        if not audio.content_type or (
            not audio.content_type.startswith('audio') and
            not audio.content_type.startswith('video')
        ):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser de tipo audio o video/webm"
            )

        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="El archivo de audio está vacío")

        transcribed_text = await transcribe_audio(audio_bytes)
        return {"text": transcribed_text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribiendo audio: {str(e)}")


class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"

@app.post("/voice/tts")
async def voice_tts(request: TTSRequest):
    """
    Convierte texto a audio (solo TTS). Endpoint rápido (~1-2s).
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

        audio_bytes = await text_to_speech(request.text, voice=request.voice)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio": audio_base64,
            "audioType": "audio/ogg"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando audio: {str(e)}")


@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    voice_name: str = "nova",
    history: str = None,
    user_profile_json: str = None,
):
    """
    Pipeline completo de voz (STT → Chatbot → TTS).
    Acepta historial y perfil de usuario para respuestas contextuales.
    """
    try:
        if not audio.content_type or (
            not audio.content_type.startswith('audio') and
            not audio.content_type.startswith('video')
        ):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser de tipo audio o video/webm"
            )

        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="El archivo de audio está vacío")

        # Parse optional history and profile from form data
        parsed_history = None
        parsed_profile = None
        if history:
            try:
                parsed_history = json.loads(history)
            except json.JSONDecodeError:
                pass
        if user_profile_json:
            try:
                parsed_profile = json.loads(user_profile_json)
            except json.JSONDecodeError:
                pass

        response_audio, transcribed_text, chatbot_response = await process_voice_message(
            audio_bytes,
            voice=voice_name,
            history=parsed_history,
            user_profile=parsed_profile
        )

        audio_base64 = base64.b64encode(response_audio).decode("utf-8")

        return {
            "text": transcribed_text,
            "response": chatbot_response,
            "audio": audio_base64,
            "audioType": "audio/ogg"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando audio: {str(e)}")
