"""
Tool registry for the OpenAI Realtime API voice path.

Mirrors the LangGraph @tool definitions in chatbot.py but exposes them as plain
async callables that take an explicit context dict, so they can be invoked from
the Realtime tool-call event handler without depending on module globals.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from news import get_spain_news, format_news_for_chat
from weather import get_weather, format_weather_for_chat
from spanish_newspapers import (
    get_combined_news,
    format_newspapers_for_chat,
    get_newspapers_by_source,
)
from alert import send_sms_alert
from spotify import get_user_data as spotify_get_user_data, format_spotify_for_chat
from reminders import create_reminder as reminders_create, list_active_reminders
from activities import search_activities


# OpenAI Realtime API tool schemas (function-tool form).
REALTIME_TOOLS: list[dict] = [
    {
        "type": "function",
        "name": "obtener_noticias",
        "description": (
            "Obtiene las noticias mas recientes de Espana desde NewsAPI. "
            "Usa cuando el usuario pregunte por noticias o actualidad."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limite": {
                    "type": "integer",
                    "description": "Numero de noticias (1-10).",
                    "default": 5,
                }
            },
        },
    },
    {
        "type": "function",
        "name": "obtener_clima",
        "description": (
            "Obtiene el clima actual de una ciudad espanola. Si no se indica "
            "ciudad, usa la del perfil del usuario."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ciudad": {
                    "type": "string",
                    "description": "Ciudad de Espana (Madrid, Barcelona...). Vacio = ciudad del perfil.",
                }
            },
        },
    },
    {
        "type": "function",
        "name": "obtener_noticias_periodicos",
        "description": (
            "Noticias directas de periodicos espanoles via RSS. Fuentes: elpais, "
            "elmundo, larazon, elperiodico, lavanguardia, abc, elespanol, "
            "elconfidencial, eldiario, mundodeportivo, o 'todos'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limite_por_fuente": {"type": "integer", "default": 3},
                "periodico": {"type": "string", "default": "todos"},
            },
        },
    },
    {
        "type": "function",
        "name": "enviar_alerta_sms",
        "description": (
            "Envia alerta SMS al tutor/cuidador. Nombre y GPS se adjuntan "
            "automaticamente. Usa solo cuando el usuario pida explicitamente "
            "enviar una alerta o describa una emergencia."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "descripcion": {
                    "type": "string",
                    "description": "Resumen breve (1-2 frases) de lo ocurrido.",
                }
            },
        },
    },
    {
        "type": "function",
        "name": "obtener_musica_spotify",
        "description": (
            "Obtiene actividad de Spotify del usuario (top artistas, "
            "recientemente escuchado, playlists). Requiere cuenta vinculada."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tipo": {
                    "type": "string",
                    "enum": ["all", "top", "recent", "playlists"],
                    "default": "all",
                }
            },
        },
    },
    {
        "type": "function",
        "name": "crear_recordatorio",
        "description": (
            "Crea un recordatorio. SIEMPRE confirma con el usuario antes de "
            "llamar. fecha_hora en ISO 8601 con offset Europe/Madrid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mensaje": {"type": "string"},
                "fecha_hora": {
                    "type": "string",
                    "description": "ISO 8601, ej '2026-04-29T15:00:00+02:00'.",
                },
                "recurrencia": {
                    "type": "string",
                    "description": "Expresion cron opcional, ej '0 9 * * *'.",
                },
            },
            "required": ["mensaje", "fecha_hora"],
        },
    },
    {
        "type": "function",
        "name": "listar_recordatorios",
        "description": "Lista recordatorios activos del usuario.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "buscar_actividades",
        "description": (
            "Busca actividades y lugares de interes para mayores cerca del usuario."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "radio_km": {"type": "integer", "default": 10},
            },
        },
    },
]


# ---------- Tool implementations ----------

async def _tool_obtener_noticias(args: dict, ctx: dict) -> str:
    limite = min(int(args.get("limite", 5) or 5), 10)
    data = await asyncio.to_thread(get_spain_news, limit=limite)
    return format_news_for_chat(data)


async def _tool_obtener_clima(args: dict, ctx: dict) -> str:
    ciudad = (args.get("ciudad") or "").strip()
    if not ciudad:
        ciudad = (ctx.get("user_profile") or {}).get("city") or "Madrid"
    data = await asyncio.to_thread(get_weather, city=ciudad, country_code="ES")
    return format_weather_for_chat(data)


async def _tool_obtener_noticias_periodicos(args: dict, ctx: dict) -> str:
    limite = int(args.get("limite_por_fuente", 3) or 3)
    periodico = (args.get("periodico") or "todos").lower()
    if periodico == "todos":
        data = await asyncio.to_thread(get_combined_news, limit_per_source=limite)
    else:
        data = await asyncio.to_thread(
            get_newspapers_by_source, source=periodico, limit=limite * 2
        )
    return format_newspapers_for_chat(data)


async def _tool_enviar_alerta_sms(args: dict, ctx: dict) -> str:
    tutor = ctx.get("tutor_profile") or {}
    tutor_number = tutor.get("number")
    if not tutor_number:
        return (
            "No se pudo enviar la alerta: no hay numero de tutor configurado. "
            "Pide al usuario que anada el contacto del cuidador."
        )
    profile = ctx.get("user_profile") or {}
    location = ctx.get("user_location") or {}
    description = (args.get("descripcion") or "").strip()[:280] or None
    result = await asyncio.to_thread(
        send_sms_alert,
        to=tutor_number,
        user_name=profile.get("name"),
        latitude=location.get("latitude"),
        longitude=location.get("longitude"),
        description=description,
    )
    if result.get("error"):
        return f"No se pudo enviar la alerta: {result['error']}"
    alert_info = result["alert"]
    return f"Alerta enviada correctamente por SMS al numero {alert_info['destino']}."


async def _tool_obtener_musica_spotify(args: dict, ctx: dict) -> str:
    user_id = ctx.get("user_id")
    if not user_id:
        return "No se pudo identificar al usuario para consultar Spotify."
    kind = args.get("tipo") or "all"
    if kind not in ("all", "top", "recent", "playlists"):
        kind = "all"
    data = await spotify_get_user_data(user_id, kind=kind)
    return format_spotify_for_chat(data)


async def _tool_crear_recordatorio(args: dict, ctx: dict) -> str:
    user_id = ctx.get("user_id")
    if not user_id:
        return "Error: no se pudo identificar al usuario."
    mensaje = args.get("mensaje") or ""
    fecha_hora = args.get("fecha_hora") or ""
    recurrencia = args.get("recurrencia") or None
    if not mensaje or not fecha_hora:
        return "Error: faltan datos del recordatorio (mensaje o fecha)."
    try:
        dt = datetime.fromisoformat(fecha_hora)
        if dt.tzinfo is None:
            madrid_tz = ZoneInfo("Europe/Madrid")
            fecha_hora = dt.replace(tzinfo=madrid_tz).astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    try:
        await reminders_create(
            user_id=user_id,
            message=mensaje,
            remind_at=fecha_hora,
            recurrence=recurrencia,
        )
        if recurrencia:
            return f"Recordatorio recurrente creado: '{mensaje}'. Proximo aviso: {fecha_hora}."
        return f"Recordatorio creado: '{mensaje}' para el {fecha_hora}."
    except Exception as e:
        return f"Error al crear el recordatorio: {str(e)}"


async def _tool_listar_recordatorios(args: dict, ctx: dict) -> str:
    user_id = ctx.get("user_id")
    if not user_id:
        return "Error: no se pudo identificar al usuario."
    try:
        items = await list_active_reminders(user_id=user_id)
        if not items:
            return "No tienes recordatorios activos en este momento."
        lines = ["Tus recordatorios activos:"]
        for r in items:
            mark = "(recurrente)" if r.get("recurrence") else ""
            lines.append(f"- {r['message']} - {r['remind_at']} {mark}".rstrip())
        return "\n".join(lines)
    except Exception as e:
        return f"Error al obtener los recordatorios: {str(e)}"


async def _tool_buscar_actividades(args: dict, ctx: dict) -> str:
    profile = ctx.get("user_profile") or {}
    tutor = ctx.get("tutor_profile") or {}
    location = ctx.get("user_location") or {}
    radio_km = int(args.get("radio_km", 10) or 10)
    return await asyncio.to_thread(
        search_activities,
        user_profile=profile,
        tutor_factors=tutor.get("factors", "") or "",
        latitude=location.get("latitude"),
        longitude=location.get("longitude"),
        radius_km=radio_km,
    )


TOOL_REGISTRY: dict[str, Callable[[dict, dict], Awaitable[str]]] = {
    "obtener_noticias": _tool_obtener_noticias,
    "obtener_clima": _tool_obtener_clima,
    "obtener_noticias_periodicos": _tool_obtener_noticias_periodicos,
    "enviar_alerta_sms": _tool_enviar_alerta_sms,
    "obtener_musica_spotify": _tool_obtener_musica_spotify,
    "crear_recordatorio": _tool_crear_recordatorio,
    "listar_recordatorios": _tool_listar_recordatorios,
    "buscar_actividades": _tool_buscar_actividades,
}


async def execute_tool(name: str, arguments: Any, context: dict) -> str:
    """Execute a tool by name with raw arguments (dict or JSON string)."""
    if name not in TOOL_REGISTRY:
        return f"Error: herramienta desconocida '{name}'."
    if isinstance(arguments, str):
        import json
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    try:
        return await TOOL_REGISTRY[name](arguments, context or {})
    except Exception as e:
        return f"Error ejecutando {name}: {str(e)}"
