import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional, List
from zoneinfo import ZoneInfo
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import websockets as websockets_client
from pydantic import BaseModel
from chatbot import chatbot_async, chatbot_stream
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
from tool_registry import REALTIME_TOOLS, execute_tool
from alert import send_sms_alert
from memory_service import run_memory_pipeline
from social_google import get_status as google_get_status, get_user_data as google_get_user_data
from spotify import get_status as spotify_get_status, get_user_data as spotify_get_user_data
from reminders import (
    list_active_reminders,
    create_reminder,
    update_reminder,
    get_unread_notifications,
    mark_notification_read,
)
from reminder_scheduler import scheduler_loop, run_tick
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
    user_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: Optional[str] = None

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
    Envía una alerta por SMS usando Twilio.

    Args:
        request: JSON con campos opcionales: to, user_name, latitude, longitude, description

    Returns:
        JSON con el resultado del envío
    """
    result = send_sms_alert(
        to=request.to,
        user_name=request.user_name,
        latitude=request.latitude,
        longitude=request.longitude,
        description=request.description,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/social/google/{user_id}/status")
async def social_google_status(user_id: str):
    """Whether the user has linked their Google account, and which scopes."""
    return await google_get_status(user_id)


@app.get("/social/google/{user_id}/data")
async def social_google_data(user_id: str, kind: str = "all"):
    """Fetch Calendar events + YouTube subscriptions for a linked user.

    `kind` filters which slice to fetch: `all`, `calendar`, `youtube`.
    """
    if kind not in ("all", "calendar", "youtube"):
        raise HTTPException(status_code=400, detail="kind inválido")
    return await google_get_user_data(user_id, kind=kind)


@app.get("/social/spotify/{user_id}/status")
async def social_spotify_status(user_id: str):
    """Whether the user has linked their Spotify account, and which scopes."""
    return await spotify_get_status(user_id)


@app.get("/social/spotify/{user_id}/data")
async def social_spotify_data(user_id: str, kind: str = "all"):
    """Fetch top artists + recently played + playlists for a linked user.

    `kind` filters which slice to fetch: `all`, `top`, `recent`, `playlists`.
    """
    if kind not in ("all", "top", "recent", "playlists"):
        raise HTTPException(status_code=400, detail="kind inválido")
    return await spotify_get_user_data(user_id, kind=kind)


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


class RealtimeSessionRequest(BaseModel):
    user_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None
    tutor_profile: Optional[TutorProfile] = None
    user_memory: Optional[dict] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class RealtimeToolRequest(BaseModel):
    name: str
    arguments: Optional[Any] = None
    user_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None
    tutor_profile: Optional[TutorProfile] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


def _clip_for_prompt(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _build_realtime_instructions(
    profile: Optional[dict],
    tutor: Optional[dict],
    user_memory: Optional[dict],
) -> str:
    madrid_now = datetime.now(ZoneInfo("Europe/Madrid"))
    tz_offset_raw = madrid_now.strftime("%z")
    tz_offset = f"{tz_offset_raw[:3]}:{tz_offset_raw[3:]}"
    lines = [
        "Eres MenteViva, un asistente de voz para personas mayores.",
        (
            "Responde siempre en espanol claro, calido y muy breve: "
            "normalmente 1-2 frases cortas."
        ),
        (
            f"Fecha y hora actual: {madrid_now.strftime('%Y-%m-%d %H:%M')} "
            f"Europe/Madrid ({tz_offset})."
        ),
        "Si el usuario te interrumpe, deja de hablar y atiende la nueva peticion.",
    ]

    if profile:
        profile_bits = []
        if profile.get("name"):
            profile_bits.append(f"nombre={_clip_for_prompt(profile.get('name'), 60)}")
        if profile.get("city"):
            profile_bits.append(f"ciudad={_clip_for_prompt(profile.get('city'), 60)}")
        if profile.get("interests"):
            profile_bits.append(f"intereses={_clip_for_prompt(profile.get('interests'), 180)}")
        if profile.get("description"):
            profile_bits.append(f"descripcion={_clip_for_prompt(profile.get('description'), 180)}")
        if profile_bits:
            lines.append("Perfil del usuario: " + "; ".join(profile_bits) + ".")

    if tutor:
        tutor_bits = []
        if tutor.get("name"):
            tutor_bits.append(f"nombre={_clip_for_prompt(tutor.get('name'), 60)}")
        if tutor.get("relationship"):
            tutor_bits.append(f"relacion={_clip_for_prompt(tutor.get('relationship'), 60)}")
        if tutor_bits:
            lines.append("Tutor/cuidador: " + "; ".join(tutor_bits) + ".")
        if tutor.get("factors"):
            lines.append(
                "Factores importantes del usuario: "
                + _clip_for_prompt(tutor.get("factors"), 260)
                + "."
            )

    if user_memory:
        narrative = _clip_for_prompt(user_memory.get("narrative"), 320)
        facts = user_memory.get("facts") or []
        fact_texts = []
        for fact in facts[:5]:
            if isinstance(fact, dict) and fact.get("text"):
                fact_texts.append(_clip_for_prompt(fact.get("text"), 90))
        if narrative:
            lines.append("Memoria resumida: " + narrative)
        if fact_texts:
            lines.append("Hechos utiles: " + "; ".join(fact_texts) + ".")

    lines.extend(
        [
            (
                "Usa herramientas solo cuando hagan falta: clima, noticias, alerta SMS, "
                "Spotify, recordatorios y actividades cercanas."
            ),
            (
                "Para recordatorios, confirma antes de crearlos y usa ISO 8601 con "
                f"offset de Madrid ({tz_offset})."
            ),
            (
                "Para emergencias o alertas, usa enviar_alerta_sms si el usuario lo "
                "pide explicitamente o describe una situacion de riesgo."
            ),
        ]
    )
    return "\n".join(lines)


XAI_REALTIME_URL = os.getenv("XAI_REALTIME_URL", "wss://api.x.ai/v1/realtime")
XAI_REALTIME_MODEL = os.getenv("XAI_REALTIME_MODEL", "grok-voice-think-fast-1.0")
XAI_REALTIME_VOICE = os.getenv("XAI_REALTIME_VOICE", "c630b236")


def _get_xai_api_key() -> Optional[str]:
    # Backward-compatible alias: some deployments still use X_API_KEY.
    return os.getenv("XAI_API_KEY") or os.getenv("X_API_KEY")


def _build_realtime_session(
    profile: Optional[dict],
    tutor: Optional[dict],
    user_memory: Optional[dict],
) -> dict:
    instructions = _build_realtime_instructions(profile, tutor, user_memory)
    return {
        "voice": XAI_REALTIME_VOICE,
        "instructions": instructions,
        "turn_detection": {"type": "server_vad"},
        "tools": REALTIME_TOOLS,
    }


def _extract_realtime_function_call(payload: dict) -> Optional[dict]:
    """
    Extract a function call from xAI/OpenAI-Realtime style events.

    Supported event shapes:
    - response.output_item.done with item.type=function_call
    - conversation.item.created with item.type=function_call
    - response.function_call_arguments.done
    """
    if not isinstance(payload, dict):
        return None

    event_type = payload.get("type")
    item = None

    if event_type in {"response.output_item.done", "conversation.item.created"}:
        candidate = payload.get("item")
        if isinstance(candidate, dict):
            item = candidate
    elif event_type == "response.function_call_arguments.done":
        name = payload.get("name")
        call_id = payload.get("call_id")
        if name and call_id:
            return {
                "name": name,
                "arguments": payload.get("arguments"),
                "call_id": call_id,
            }

    if not isinstance(item, dict) or item.get("type") != "function_call":
        return None

    name = item.get("name")
    call_id = item.get("call_id")
    if not name or not call_id:
        return None

    return {
        "name": name,
        "arguments": item.get("arguments"),
        "call_id": call_id,
    }


async def _send_realtime_tool_result(
    xai_ws: Any,
    tool_call: dict,
    context: dict,
) -> None:
    result = await execute_tool(
        str(tool_call["name"]),
        tool_call.get("arguments"),
        context,
    )
    await xai_ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": str(tool_call["call_id"]),
                    "output": result,
                },
            }
        )
    )
    await xai_ws.send(json.dumps({"type": "response.create"}))


@app.websocket("/realtime/ws")
async def realtime_ws(ws: WebSocket):
    """
    Bidirectional relay between the browser and xAI's Realtime WebSocket API.

    Flow:
      1. Browser connects, sends one JSON init message with user context.
      2. Backend dials wss://api.x.ai/v1/realtime?model=...
      3. Backend sends session.update (instructions + tools) to xAI.
      4. By default, backend executes function calls and sends
         function_call_output automatically.
      5. If init message includes {"tool_call_handler":"frontend"}, backend
         only relays and frontend can handle tools via POST /realtime/tool.
    """
    await ws.accept()
    api_key = _get_xai_api_key()
    if not api_key:
        await ws.send_json(
            {
                "type": "error",
                "error": {"message": "XAI_API_KEY no configurada (o X_API_KEY)"},
            }
        )
        await ws.close()
        return

    try:
        init_msg = await ws.receive_json()
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_json(
            {"type": "error", "error": {"message": f"init invalido: {e}"}}
        )
        await ws.close()
        return

    profile = init_msg.get("user_profile")
    if not isinstance(profile, dict):
        profile = {}

    tutor = init_msg.get("tutor_profile")
    if not isinstance(tutor, dict):
        tutor = {}

    memory = init_msg.get("user_memory")
    user_id = init_msg.get("user_id") or profile.get("id")
    if user_id and not profile.get("id"):
        profile["id"] = user_id

    user_location = {}
    latitude = init_msg.get("latitude")
    longitude = init_msg.get("longitude")
    if latitude is not None and longitude is not None:
        user_location = {"latitude": latitude, "longitude": longitude}

    tool_call_handler = str(init_msg.get("tool_call_handler") or "backend").lower()
    backend_handles_tools = tool_call_handler != "frontend"
    tool_context = {
        "user_id": user_id,
        "user_profile": profile,
        "tutor_profile": tutor,
        "user_location": user_location,
    }
    handled_tool_call_ids: set[str] = set()

    session_config = _build_realtime_session(profile, tutor, memory)

    xai_url = f"{XAI_REALTIME_URL}?model={XAI_REALTIME_MODEL}"
    try:
        async with websockets_client.connect(
            xai_url,
            additional_headers={"Authorization": f"Bearer {api_key}"},
            max_size=16 * 1024 * 1024,
            ping_interval=20,
        ) as xai_ws:
            await xai_ws.send(
                json.dumps({"type": "session.update", "session": session_config})
            )
            await ws.send_json({"type": "relay.ready"})

            async def client_to_xai():
                try:
                    while True:
                        msg = await ws.receive_text()
                        await xai_ws.send(msg)
                except WebSocketDisconnect:
                    return
                except Exception:
                    return

            async def xai_to_client():
                try:
                    async for msg in xai_ws:
                        if isinstance(msg, bytes):
                            msg = msg.decode("utf-8", errors="ignore")

                        if backend_handles_tools:
                            try:
                                payload = json.loads(msg)
                            except json.JSONDecodeError:
                                payload = None

                            if isinstance(payload, dict):
                                tool_call = _extract_realtime_function_call(payload)
                                if tool_call:
                                    call_id = str(tool_call["call_id"])
                                    if call_id not in handled_tool_call_ids:
                                        handled_tool_call_ids.add(call_id)
                                        await _send_realtime_tool_result(
                                            xai_ws=xai_ws,
                                            tool_call=tool_call,
                                            context=tool_context,
                                        )
                        await ws.send_text(msg)
                except websockets_client.ConnectionClosed:
                    return
                except Exception:
                    return

            done, pending = await asyncio.wait(
                {
                    asyncio.create_task(client_to_xai()),
                    asyncio.create_task(xai_to_client()),
                },
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "error": {"message": str(e)}})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.post("/realtime/session")
async def realtime_session(request: RealtimeSessionRequest):
    """
    Returns the metadata the browser needs to open the realtime WebSocket.
    Kept as POST so the same /api/realtime/session Next.js proxy keeps working;
    the response shape changed from SDP to a small JSON config.
    """
    if not _get_xai_api_key():
        raise HTTPException(
            status_code=500, detail="XAI_API_KEY no configurada (o X_API_KEY)"
        )
    return {
        "ws_path": "/realtime/ws",
        "model": XAI_REALTIME_MODEL,
        "voice": XAI_REALTIME_VOICE,
        "tool_call_handler": "backend",
        "input_sample_rate": 24000,
        "output_sample_rate": 24000,
    }


@app.post("/realtime/tool")
async def realtime_tool(request: RealtimeToolRequest):
    """
    Execute a tool by name. Called by the frontend whenever the Realtime model
    emits a function_call event. Result is sent back to the model as a
    conversation.item.create with type=function_call_output.
    """
    profile = request.user_profile.model_dump() if request.user_profile else {}
    if request.user_id:
        profile["id"] = request.user_id
    tutor = request.tutor_profile.model_dump() if request.tutor_profile else {}
    user_location = {}
    if request.latitude is not None and request.longitude is not None:
        user_location = {"latitude": request.latitude, "longitude": request.longitude}

    context = {
        "user_id": request.user_id,
        "user_profile": profile,
        "tutor_profile": tutor,
        "user_location": user_location,
    }
    result = await execute_tool(request.name, request.arguments, context)
    return {"result": result}


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
