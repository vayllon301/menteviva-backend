from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from datetime import datetime
from news import get_spain_news, format_news_for_chat
from weather import get_weather, format_weather_for_chat
from spanish_newspapers import get_combined_news, format_newspapers_for_chat, get_newspapers_by_source

load_dotenv()

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
def obtener_noticias_periodicos(limite_por_fuente: int = 5, periodico: str = "ambos") -> str:
    """
    Obtiene las noticias más recientes directamente de los periódicos El País y El Mundo.
    Esta herramienta accede a las fuentes originales de estos periódicos.
    Usa esta herramienta cuando el usuario específicamente pida noticias de estos periódicos
    o cuando quiera noticias actualizadas de fuentes confiables españolas.
    
    Args:
        limite_por_fuente: Número de noticias a obtener de cada periódico (por defecto 5)
        periodico: Qué periódico consultar: "ambos" (defecto), "elpais", o "elmundo"
    
    Returns:
        Noticias actualizadas formateadas con la fecha de hoy, listas para presentar al usuario
    """
    if periodico.lower() == "ambos":
        news_data = get_combined_news(limit_per_source=limite_por_fuente)
    else:
        news_data = get_newspapers_by_source(source=periodico, limit=limite_por_fuente * 2)
    
    return format_newspapers_for_chat(news_data)

# Define the list of tools
tools = [obtener_noticias, obtener_clima, obtener_noticias_periodicos]

class State(TypedDict):
    # Messages are appended to the list using add_messages
    messages: Annotated[list, add_messages]

def chatbot_node(state: State):
    # Initialize the LLM with the API key from environment
    # langchain-openai looks for OPENAI_API_KEY by default
    llm = ChatOpenAI(model="gpt-5-nano")
    
    # Bind tools to the LLM so it knows it can call them
    llm_with_tools = llm.bind_tools(tools)
    
    # Obtener la fecha actual en español
    today = datetime.now()
    days_es = {
        0: "Lunes", 1: "Martes", 2: "Miércoles",
        3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"
    }
    months_es = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    today_str = f"{days_es[today.weekday()]}, {today.day} de {months_es[today.month]} de {today.year}"
    
    system_message = {
        "role": "system",
        "content": (
            f"FECHA ACTUAL: {today_str}\n\n"
            "Eres MenteViva, un asistente de IA paciente, respetuoso y cálido, diseñado específicamente para ayudar a personas mayores. "
            "Tu objetivo es brindar compañía, ayudar con las tareas diarias y fomentar la salud cognitiva.\n\n"
            "Al interactuar:\n"
            "1. Usa un lenguaje claro y sencillo, evita tecnicismos.\n"
            "2. Sé extremadamente paciente y alentador.\n"
            "3. Si el usuario parece confundido, ofrece orientación amable.\n"
            "4. Habla con un tono cálido y respetuoso. Usa un trato formal y educado.\n"
            "5. Ofrece recordatorios de hábitos saludables como beber agua, dar un paseo corto o hacer un rompecabezas.\n"
            "6. Si te preguntan sobre consejos médicos, recuerda siempre consultar con su médico o un profesional.\n"
            "7. Mantén las respuestas concisas pero amigables para no abrumar al usuario.\n\n"
            "CUANDO EL USUARIO PREGUNTA SOBRE EL CLIMA:\n"
            "- Llama a la herramienta obtener_clima para obtener los datos actuales.\n"
            "- Transforma los datos técnicos en un lenguaje amable y práctico.\n"
            "- Destaca lo más importante: temperatura actual, condiciones generales, y recomendaciones útiles.\n"
            "- Incluye consejos prácticos: qué ropa usar, si llevar paraguas, si es buen día para pasear, etc.\n"
            "- Sé conciso: no abrumes con todos los datos técnicos (presión, nubosidad, etc.).\n"
            "- Ejemplo: En lugar de listar todos los números, di algo como: 'Hace 18 grados y está parcialmente nublado. Sería un buen día para dar un paseo, pero lleva un abrigo ligero.'\n\n"
            "CUANDO EL USUARIO PREGUNTA SOBRE NOTICIAS:\n"
            "- Si pide noticias de El País, El Mundo, o de periódicos específicos, usa la herramienta obtener_noticias_periodicos.\n"
            "- Si pide noticias generales de España, puedes usar obtener_noticias o obtener_noticias_periodicos.\n"
            "- Las noticias de obtener_noticias_periodicos son directamente de las fuentes originales y están actualizadas.\n"
            "- Siempre menciona que las noticias son del día de hoy para dar contexto temporal.\n\n"
            "HERRAMIENTAS DISPONIBLES:\n"
            "- obtener_noticias: Noticias generales de España desde NewsAPI\n"
            "- obtener_noticias_periodicos: Noticias directas de El País y El Mundo (RSS feeds actualizados)\n"
            "- obtener_clima: Clima actual de cualquier ciudad de España\n"
            "- Usa estas herramientas de manera proactiva cuando sea apropiado para ayudar al usuario.\n"
        )
    }
    
    # Prepend system message to the conversation history
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

def chatbot(message: str):
    # Helper function to interface with main.py
    input_state = {"messages": [("user", message)]}
    result = graph.invoke(input_state)
    return result["messages"][-1].content
