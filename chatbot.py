from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from datetime import datetime
from news import get_spain_news, format_news_for_chat
from weather import get_weather, format_weather_for_chat
from spanish_newspapers import get_combined_news, format_newspapers_for_chat, get_newspapers_by_source
from alert import send_whatsapp_alert
from instagram import get_instagram_info, format_instagram_for_chat
from reminders import create_reminder, list_active_reminders
from activities import search_activities

load_dotenv()

DAYS_ES = {
    0: "Lunes", 1: "Martes", 2: "Miércoles",
    3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"
}
MONTHS_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

# Define tools for the chatbot to use
@tool
def obtener_noticias(limite: int = 5) -> str:
    """
    Obtiene las noticias más recientes de España. Usa esta herramienta cuando el usuario
    pregunte sobre noticias, actualidad, lo que está pasando hoy, o información reciente.

    Args:
        limite: Número de noticias a obtener (por defecto 5, máximo recomendado 10)

    Returns:
        Noticias formateadas listas para presentar al usuario
    """
    limite = min(limite, 10)  # Limitar para no abrumar al usuario
    news_data = get_spain_news(limit=limite)
    return format_news_for_chat(news_data)

@tool
def obtener_clima(ciudad: str = "Barcelona") -> str:
    """
    Obtiene el clima actual de una ciudad en España. Usa esta herramienta cuando el usuario
    pregunte sobre el tiempo, el clima, la temperatura o las condiciones meteorológicas.

    Args:
        ciudad: Nombre de la ciudad (por defecto "Barcelona"). El usuario puede especificar
                cualquier ciudad de España como Barcelona, Valencia, Sevilla, etc.

    Returns:
        Información del clima formateada lista para presentar al usuario
    """
    weather_data = get_weather(city=ciudad, country_code="ES")
    return format_weather_for_chat(weather_data)

@tool
def obtener_noticias_periodicos(limite_por_fuente: int = 3, periodico: str = "todos") -> str:
    """
    Obtiene las noticias más recientes directamente de periódicos españoles.
    Fuentes disponibles: El País, El Mundo, La Razón, El Periódico, La Vanguardia,
    ABC, El Español, El Confidencial, eldiario.es, Mundo Deportivo.
    Usa esta herramienta cuando el usuario pida noticias de periódicos españoles.

    Args:
        limite_por_fuente: Número de noticias a obtener de cada periódico (por defecto 3)
        periodico: Qué periódico consultar: "todos" (defecto), "elpais", "elmundo",
                   "larazon", "elperiodico", "lavanguardia", "abc", "elespanol",
                   "elconfidencial", "eldiario", "mundodeportivo"

    Returns:
        Noticias actualizadas formateadas con la fecha de hoy, listas para presentar al usuario
    """
    if periodico.lower() == "todos":
        news_data = get_combined_news(limit_per_source=limite_por_fuente)
    else:
        news_data = get_newspapers_by_source(source=periodico, limit=limite_por_fuente * 2)

    return format_newspapers_for_chat(news_data)

@tool
def enviar_alerta_whatsapp() -> str:
    """
    Envía una alerta predefinida por WhatsApp al cuidador o familiar del usuario.
    Usa esta herramienta cuando el usuario pida enviar una alerta o aviso por WhatsApp,
    o cuando detectes una situación que requiera notificar a alguien (emergencia, recordatorio importante, etc.).

    Returns:
        Confirmación del envío o mensaje de error
    """
    result = send_whatsapp_alert()
    if result.get("error"):
        return f"No se pudo enviar la alerta: {result['error']}"
    alert_info = result["alert"]
    return f"Alerta enviada correctamente por WhatsApp al número {alert_info['destino']}."

@tool
def obtener_instagram(usuario: str) -> str:
    """
    Obtiene información pública del perfil de Instagram de una persona.
    Usa esta herramienta cuando el usuario pregunte sobre la cuenta de Instagram
    de su cuidador/tutor, o cuando quiera saber información de un perfil de Instagram.
    El nombre de usuario del tutor aparece en el perfil del tutor si está vinculado.

    Args:
        usuario: Nombre de usuario de Instagram (sin @). Por ejemplo: "juan.perez"

    Returns:
        Información del perfil de Instagram: nombre, biografía, seguidores, publicaciones, etc.
    """
    info = get_instagram_info(usuario)
    return format_instagram_for_chat(info)

@tool
def crear_recordatorio(mensaje: str, fecha_hora: str, recurrencia: str = "") -> str:
    """
    Crea un recordatorio para el usuario. IMPORTANTE: SIEMPRE pide confirmación
    al usuario antes de llamar a esta herramienta.

    Args:
        mensaje: Texto del recordatorio (ej: "Tomar la pastilla")
        fecha_hora: Fecha y hora en formato ISO 8601 (ej: "2026-03-29T15:00:00")
        recurrencia: Expresión cron para recordatorios recurrentes (opcional).
                     Ejemplos: "0 */2 * * *" (cada 2 horas), "0 9 * * *" (cada día a las 9),
                     "0 9,14,21 * * *" (a las 9, 14 y 21h). Dejar vacío para un solo recordatorio.

    Returns:
        Confirmación del recordatorio creado
    """
    import asyncio
    import concurrent.futures

    recurrence = recurrencia if recurrencia else None
    user_id = _current_user_id

    if not user_id:
        return "Error: no se pudo identificar al usuario. Inténtalo de nuevo."

    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(
                asyncio.run,
                create_reminder(
                    user_id=user_id,
                    message=mensaje,
                    remind_at=fecha_hora,
                    recurrence=recurrence,
                )
            ).result()
        if recurrence:
            return f"Recordatorio recurrente creado: '{mensaje}'. Próximo aviso: {fecha_hora}."
        else:
            return f"Recordatorio creado: '{mensaje}' para el {fecha_hora}."
    except Exception as e:
        return f"Error al crear el recordatorio: {str(e)}"


@tool
def listar_recordatorios() -> str:
    """
    Lista los recordatorios activos del usuario. Usa esta herramienta cuando
    el usuario pregunte qué recordatorios tiene, o quiera ver sus recordatorios.

    Returns:
        Lista formateada de recordatorios activos
    """
    import asyncio
    import concurrent.futures

    user_id = _current_user_id

    if not user_id:
        return "Error: no se pudo identificar al usuario."

    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            reminders_list = pool.submit(
                asyncio.run, list_active_reminders(user_id=user_id)
            ).result()

        if not reminders_list:
            return "No tienes recordatorios activos en este momento."

        lines = ["Tus recordatorios activos:\n"]
        for r in reminders_list:
            status_emoji = "🔁" if r.get("recurrence") else "⏰"
            lines.append(
                f"{status_emoji} {r['message']} — {r['remind_at']}"
                + (f" (recurrente)" if r.get("recurrence") else "")
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error al obtener los recordatorios: {str(e)}"


@tool
def buscar_actividades(radio_km: int = 10) -> str:
    """
    Busca actividades y lugares de interes para personas mayores cerca
    de la ubicacion del usuario. Usa esta herramienta cuando el usuario
    pregunte por actividades, cosas que hacer, planes, talleres, centros
    de mayores o lugares para visitar en su zona.

    Args:
        radio_km: Radio de busqueda en kilometros (por defecto 10)

    Returns:
        Lista de actividades personalizadas cerca del usuario
    """
    return search_activities(
        user_profile=_current_user_profile,
        tutor_factors=_current_tutor_factors,
        latitude=_current_user_location.get("latitude"),
        longitude=_current_user_location.get("longitude"),
        radius_km=radio_km,
    )


# Define the list of tools
tools = [
    obtener_noticias,
    obtener_clima,
    obtener_noticias_periodicos,
    enviar_alerta_whatsapp,
    obtener_instagram,
    crear_recordatorio,
    listar_recordatorios,
    buscar_actividades,
]

_current_user_id = ""
_current_user_location = {}
_current_user_profile = {}
_current_tutor_factors = ""

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_profile: dict
    tutor_profile: dict
    user_memory: dict
    user_location: dict

def build_system_message(user_profile: dict = None, tutor_profile: dict = None, user_memory: dict = None):
    """Build the system message, optionally personalized with user and tutor profiles."""
    today = datetime.now()
    today_str = f"{DAYS_ES[today.weekday()]}, {today.day} de {MONTHS_ES[today.month]} de {today.year}"

    profile_section = ""
    if user_profile:
        profile_section = (
            f"\nPERFIL DEL USUARIO:\n"
            f"- Nombre: {user_profile.get('name', 'Desconocido')}\n"
            f"- Teléfono: {user_profile.get('number', '')}\n"
            f"- Descripción: {user_profile.get('description', '')}\n"
            f"- Intereses: {user_profile.get('interests', '')}\n"
            f"- Ciudad: {user_profile.get('city', '')}\n\n"
            "INSTRUCCIONES SOBRE EL PERFIL:\n"
            f"- Dirígete al usuario por su nombre ({user_profile.get('name', '')}) de forma natural y cálida.\n"
            f"- Cuando pregunte por el clima sin especificar ciudad, usa su ciudad ({user_profile.get('city', '')}).\n"
            "- Ten en cuenta sus intereses y descripción para personalizar tus respuestas y sugerencias.\n"
            "- Usa su número de teléfono solo si necesitas enviarlo en una alerta de WhatsApp.\n\n"
        )

    tutor_section = ""
    if tutor_profile:
        tutor_name = tutor_profile.get('name', 'Desconocido')
        tutor_number = tutor_profile.get('number', '')
        tutor_desc = tutor_profile.get('description', '')
        tutor_ig = tutor_profile.get('instagram', '')
        tutor_fb = tutor_profile.get('facebook', '')
        tutor_relationship = tutor_profile.get('relationship', '')
        tutor_factors = tutor_profile.get('factors', '')

        tutor_section = (
            f"PERFIL DEL TUTOR/CUIDADOR:\n"
            f"- Nombre: {tutor_name}\n"
            f"- Relación con el usuario: {tutor_relationship}\n"
            f"- Teléfono: {tutor_number}\n"
            f"- Descripción: {tutor_desc}\n"
            f"- Instagram: {tutor_ig}\n"
            f"- Facebook: {tutor_fb}\n\n"
            "INSTRUCCIONES SOBRE EL TUTOR:\n"
            f"- El tutor/cuidador del usuario se llama {tutor_name}"
            + (f" y es su {tutor_relationship}" if tutor_relationship else "") + ".\n"
            "- Si el usuario pregunta por su cuidador o familiar responsable, puedes referirte a esta persona.\n"
            f"- Al enviar alertas de WhatsApp, el destinatario es el tutor ({tutor_name}).\n"
            "- No compartas los datos del tutor (teléfono, redes sociales) directamente con el usuario a menos que lo solicite.\n"
            + (f"- El tutor tiene vinculada su cuenta de Instagram: @{tutor_ig}. "
               "Puedes usar la herramienta obtener_instagram para consultar su perfil si el usuario pregunta.\n" if tutor_ig else "")
            + "\n"
        )

        if tutor_factors:
            tutor_section += (
                "⚠️ FACTORES IMPORTANTES SOBRE EL USUARIO ⚠️\n"
                f"El cuidador ha indicado los siguientes factores a tener muy en cuenta:\n"
                f"{tutor_factors}\n\n"
                "INSTRUCCIONES OBLIGATORIAS SOBRE ESTOS FACTORES:\n"
                "- Adapta SIEMPRE tus respuestas y sugerencias teniendo en cuenta estos factores.\n"
                "- Nunca recomiendes actividades, alimentos o hábitos que puedan ser perjudiciales dado su estado.\n"
                "- Si el usuario menciona síntomas o situaciones relacionadas con estos factores, trátalo con especial cuidado y sugiere contactar a su médico o cuidador.\n"
                "- Si detectas una situación de riesgo relacionada con estos factores, ofrece proactivamente enviar una alerta al cuidador.\n\n"
            )

    content = (
        f"FECHA ACTUAL: {today_str}\n\n"
        "Eres MenteViva, un asistente de IA paciente, respetuoso y cálido, diseñado específicamente para ayudar a personas mayores. "
        "Tu objetivo es brindar compañía, ayudar con las tareas diarias y fomentar la salud cognitiva.\n\n"
        f"{profile_section}"
        f"{tutor_section}"
        "Al interactuar:\n"
        "1. Usa un lenguaje claro y sencillo, evita tecnicismos.\n"
        "2. Sé extremadamente paciente y alentador.\n"
        "3. Si el usuario parece confundido, ofrece orientación amable.\n"
        "4. Habla con un tono cálido y respetuoso. Usa un trato formal y educado.\n"
        "5. Ofrece recordatorios de hábitos saludables como beber agua, dar un paseo corto o hacer un rompecabezas.\n"
        "6. Si te preguntan sobre consejos médicos, recuerda siempre consultar con su médico o un profesional.\n"
        "7. IMPORTANTE - Sé BREVE y CONCISO: responde en 1-3 frases cortas siempre que sea posible. "
        "Evita párrafos largos, listas extensas y explicaciones innecesarias. "
        "Ve directo al punto. Solo extiéndete si el usuario pide más detalle o la pregunta lo requiere claramente. "
        "Recuerda que el usuario puede sentirse abrumado con textos largos.\n\n"
        "CUANDO EL USUARIO PREGUNTA SOBRE EL CLIMA:\n"
        "- Llama a la herramienta obtener_clima para obtener los datos actuales.\n"
        "- Transforma los datos técnicos en un lenguaje amable y práctico.\n"
        "- Destaca lo más importante: temperatura actual, condiciones generales, y recomendaciones útiles.\n"
        "- Incluye consejos prácticos: qué ropa usar, si llevar paraguas, si es buen día para pasear, etc.\n"
        "- Sé conciso: no abrumes con todos los datos técnicos (presión, nubosidad, etc.).\n"
        "- Ejemplo: En lugar de listar todos los números, di algo como: 'Hace 18 grados y está parcialmente nublado. Sería un buen día para dar un paseo, pero lleva un abrigo ligero.'\n\n"
        "CUANDO EL USUARIO PREGUNTA SOBRE NOTICIAS:\n"
        "- Si pide noticias de un periódico específico, usa obtener_noticias_periodicos con el nombre del periódico.\n"
        "- Periódicos disponibles: El País, El Mundo, La Razón, El Periódico, La Vanguardia, ABC, El Español, El Confidencial, eldiario.es, Mundo Deportivo.\n"
        "- Claves: elpais, elmundo, larazon, elperiodico, lavanguardia, abc, elespanol, elconfidencial, eldiario, mundodeportivo.\n"
        "- Si pide noticias generales, puedes usar obtener_noticias o obtener_noticias_periodicos con 'todos'.\n"
        "- Las noticias de obtener_noticias_periodicos son directamente de las fuentes originales y están actualizadas.\n"
        "- Siempre menciona que las noticias son del día de hoy para dar contexto temporal.\n\n"
        "CUANDO EL USUARIO PIDE ENVIAR UNA ALERTA O MENSAJE POR WHATSAPP:\n"
        "- Usa la herramienta enviar_alerta_whatsapp para enviar el mensaje.\n"
        "- Confirma al usuario que el mensaje ha sido enviado correctamente.\n"
        "- Si hay un error, informa al usuario de forma amable y sugiere intentarlo de nuevo.\n\n"
        "CUANDO EL USUARIO PREGUNTA SOBRE INSTAGRAM:\n"
        "- Si el usuario pregunta por el Instagram de su cuidador/tutor/familiar, usa la herramienta obtener_instagram con el usuario del tutor.\n"
        "- Si pregunta por cualquier otro perfil de Instagram, también puedes usar la herramienta.\n"
        "- Presenta la información de forma clara y amigable: nombre, biografía, número de seguidores y publicaciones.\n"
        "- Si el perfil es privado o no se puede acceder, explícalo amablemente.\n\n"
        "CUANDO EL USUARIO PIDE UN RECORDATORIO:\n"
        "- SIEMPRE confirma con el usuario antes de crear el recordatorio.\n"
        "- Ejemplo: 'Voy a crear un recordatorio para las 15:00: tomar la pastilla. ¿Te parece bien?'\n"
        "- Solo llama a crear_recordatorio DESPUÉS de que el usuario confirme.\n"
        "- Convierte las horas que diga el usuario a formato ISO 8601 usando la fecha actual.\n"
        "- Para recordatorios recurrentes, convierte a expresión cron:\n"
        "  - 'cada 2 horas' → '0 */2 * * *'\n"
        "  - 'todos los días a las 9' → '0 9 * * *'\n"
        "  - 'cada día a las 9, 14 y 21' → '0 9,14,21 * * *'\n"
        "- Si el usuario pregunta por sus recordatorios, usa listar_recordatorios.\n\n"
        "CUANDO EL USUARIO PREGUNTE POR ACTIVIDADES O COSAS QUE HACER:\n"
        "- Usa la herramienta buscar_actividades para encontrar lugares y actividades cerca del usuario.\n"
        "- Si usas buscar_actividades, muestra 3 a 5 resultados en lista breve.\n"
        "- Incluye para cada resultado: nombre, direccion, valoracion si existe y una recomendacion corta.\n"
        "- Presenta los resultados de forma calida y personalizada, sin resumir toda la lista en una sola frase.\n"
        "- Si no hay resultados, ofrece buscar en un radio mas amplio.\n\n"
        "HERRAMIENTAS DISPONIBLES:\n"
        "- obtener_noticias: Noticias generales de España desde NewsAPI\n"
        "- obtener_noticias_periodicos: Noticias directas de 10 periódicos españoles (RSS feeds actualizados)\n"
        "- obtener_clima: Clima actual de cualquier ciudad de España\n"
        "- enviar_alerta_whatsapp: Envía una alerta o mensaje por WhatsApp al cuidador o familiar\n"
        "- obtener_instagram: Obtiene información pública de un perfil de Instagram (seguidores, biografía, publicaciones)\n"
        "- crear_recordatorio: Crea un recordatorio para el usuario (siempre confirmar antes)\n"
        "- listar_recordatorios: Lista los recordatorios activos del usuario\n"
        "- buscar_actividades: Busca actividades y lugares de interes para mayores cerca del usuario\n"
        "- Usa estas herramientas de manera proactiva cuando sea apropiado para ayudar al usuario.\n"
    )

    memory_section = ""
    if user_memory:
        facts = user_memory.get("facts", [])
        narrative = user_memory.get("narrative", "")
        if facts or narrative:
            facts_text = "\n".join(f"- {f['text']}" for f in facts) if facts else ""
            memory_section = (
                "\nMEMORIA DEL USUARIO:\n"
                "Lo que he aprendido sobre ti a lo largo del tiempo:\n\n"
                + (f"HECHOS CONOCIDOS:\n{facts_text}\n\n" if facts_text else "")
                + (f"RESUMEN RECIENTE:\n{narrative}\n\n" if narrative else "")
                + "INSTRUCCIONES: Usa esta información de forma natural cuando sea relevante. "
                "No la menciones directamente a menos que el usuario lo haga primero.\n"
            )

    content = content + memory_section

    return {"role": "system", "content": content}

# Module-level LLM instance (avoid recreating on every request)
llm = ChatOpenAI(model="gpt-5.4", api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: State):
    global _current_user_id, _current_user_location, _current_user_profile, _current_tutor_factors
    profile = state.get("user_profile")
    if profile and profile.get("id"):
        _current_user_id = profile["id"]

    _current_user_location = state.get("user_location") or {}
    _current_user_profile = profile or {}
    tutor = state.get("tutor_profile") or {}
    _current_tutor_factors = tutor.get("factors", "") or ""

    last_message = state["messages"][-1] if state.get("messages") else None
    if isinstance(last_message, ToolMessage) and last_message.name == "buscar_actividades":
        return {"messages": [AIMessage(content=last_message.content)]}

    system_message = build_system_message(state.get("user_profile"), state.get("tutor_profile"), state.get("user_memory"))
    messages = [system_message] + state["messages"]

    return {"messages": [llm_with_tools.invoke(messages)]}

def should_continue(state: State):
    """Decide if we should continue to tools or end"""
    messages = state["messages"]
    last_message = messages[-1]

    # If there are no tool calls, we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise, we continue to the tools node
    return "continue"

# Create the tool node that will execute the tools
tool_node = ToolNode(tools)

# Create the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", tool_node)

# Add edges
workflow.add_edge(START, "chatbot")

# Add conditional edges: after chatbot, either go to tools or end
workflow.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

# After tools are executed, go back to chatbot to process the results
workflow.add_edge("tools", "chatbot")

# Compile the graph
graph = workflow.compile()

def _extract_text(content) -> str:
    """Extract plain text from LLM response content (handles Gemini's list-of-parts format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _build_messages(message: str, history: list = None):
    """Build conversation messages from history and current message."""
    messages = []
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                messages.append((role, content))
    messages.append(("user", message))
    return messages


def chatbot(message: str, history: list = None, user_profile: dict = None, tutor_profile: dict = None, user_memory: dict = None, user_location: dict = None):
    messages = _build_messages(message, history)
    input_state = {"messages": messages, "user_profile": user_profile, "tutor_profile": tutor_profile, "user_memory": user_memory, "user_location": user_location or {}}
    result = graph.invoke(input_state)
    return _extract_text(result["messages"][-1].content)


async def chatbot_async(message: str, history: list = None, user_profile: dict = None, tutor_profile: dict = None, user_memory: dict = None, user_location: dict = None):
    """Async version of chatbot using ainvoke (non-blocking)."""
    messages = _build_messages(message, history)
    input_state = {"messages": messages, "user_profile": user_profile, "tutor_profile": tutor_profile, "user_memory": user_memory, "user_location": user_location or {}}
    result = await graph.ainvoke(input_state)
    return _extract_text(result["messages"][-1].content)


async def chatbot_stream(message: str, history: list = None, user_profile: dict = None, tutor_profile: dict = None, user_memory: dict = None, user_location: dict = None):
    """Async generator that yields tokens as they are produced by the LLM."""
    messages = _build_messages(message, history)
    input_state = {"messages": messages, "user_profile": user_profile, "tutor_profile": tutor_profile, "user_memory": user_memory, "user_location": user_location or {}}
    streamed_text = False

    async for event in graph.astream_events(input_state, version="v2"):
        kind = event.get("event")
        # Stream tokens from the chatbot node's LLM calls
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                text = _extract_text(chunk.content)
                if text:
                    streamed_text = True
                    yield text
        elif kind == "on_chain_end" and event.get("name") == "chatbot" and not streamed_text:
            output = event.get("data", {}).get("output", {})
            output_messages = output.get("messages", []) if isinstance(output, dict) else []
            if output_messages:
                content = getattr(output_messages[-1], "content", "")
                text = _extract_text(content)
                if text:
                    streamed_text = True
                    yield text
