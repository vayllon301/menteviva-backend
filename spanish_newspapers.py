"""
Módulo para obtener noticias de periódicos españoles usando sus RSS feeds.
"""
import feedparser
import subprocess
from datetime import datetime
from typing import List, Dict
from bs4 import BeautifulSoup

# URLs de los RSS feeds
RSS_SOURCES = {
    "elpais": {"url": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada", "name": "El País"},
    "elmundo": {"url": "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml", "name": "El Mundo"},
    "larazon": {"url": "https://www.larazon.es/?outputType=xml", "name": "La Razón"},
    # El Periódico RSS está caído actualmente (devuelve 404)
    # "elperiodico": {"url": "https://www.elperiodico.com/es/rss/rss_portada.xml", "name": "El Periódico"},
    "lavanguardia": {"url": "https://www.lavanguardia.com/rss/home.xml", "name": "La Vanguardia"},
    "abc": {"url": "https://www.abc.es/rss/2.0/portada/", "name": "ABC"},
    "elespanol": {"url": "https://www.elespanol.com/rss/", "name": "El Español"},
    "elconfidencial": {"url": "https://rss.elconfidencial.com/", "name": "El Confidencial"},
    "eldiario": {"url": "https://www.eldiario.es/rss/", "name": "eldiario.es"},
    "mundodeportivo": {"url": "https://www.mundodeportivo.com/rss/home.xml", "name": "Mundo Deportivo"},
}

def _fetch_rss(rss_url: str):
    """Fetch and parse an RSS feed, falling back to curl for problematic servers."""
    try:
        feed = feedparser.parse(rss_url)
        if feed.entries:
            return feed
    except Exception:
        pass
    # Fallback: use curl (handles excessive headers/cookies that Python rejects)
    try:
        result = subprocess.run(
            ["curl", "-s", "-L", "--max-time", "10", rss_url],
            capture_output=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout:
            return feedparser.parse(result.stdout)
    except Exception:
        pass
    return None


def _get_rss_news(rss_url: str, source_name: str, limit: int = 5) -> List[Dict]:
    """Obtiene noticias de un RSS feed."""
    try:
        feed = _fetch_rss(rss_url)
        if not feed or not feed.entries:
            return []
        return [
            {
                "titulo": entry.get("title", "Sin título"),
                "descripcion": entry.get("summary", ""),
                "url": entry.get("link", ""),
                "fecha": entry.get("published", ""),
                "fuente": source_name,
            }
            for entry in feed.entries[:limit]
        ]
    except Exception as e:
        print(f"Error obteniendo noticias de {source_name}: {str(e)}")
        return []


def get_news_by_source(source_key: str, limit: int = 5) -> List[Dict]:
    """Obtiene noticias de un periódico por su clave en RSS_SOURCES."""
    source = RSS_SOURCES.get(source_key)
    if not source:
        return []
    return _get_rss_news(source["url"], source["name"], limit)


def get_combined_news(limit_per_source: int = 3, sources: List[str] = None) -> Dict:
    """
    Obtiene noticias combinadas de los periódicos españoles.

    Args:
        limit_per_source: Número de noticias a obtener de cada fuente
        sources: Lista de claves de periódicos a consultar. Si es None, consulta todos.

    Returns:
        Diccionario con las noticias de los periódicos
    """
    source_keys = sources if sources else list(RSS_SOURCES.keys())

    all_news = []
    for key in source_keys:
        all_news.extend(get_news_by_source(key, limit=limit_per_source))

    # Ordenar por fecha si es posible
    try:
        all_news.sort(
            key=lambda x: datetime.strptime(x['fecha'], '%a, %d %b %Y %H:%M:%S %z') if x['fecha'] else datetime.min,
            reverse=True
        )
    except (ValueError, TypeError):
        pass

    return {
        "total": len(all_news),
        "fecha_consulta": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "news": all_news
    }

def format_newspapers_for_chat(news_data: Dict) -> str:
    """
    Formatea las noticias de los periódicos en un formato legible para el chatbot.
    """
    if news_data.get("error"):
        return f"Lo siento, no pude obtener las noticias. {news_data['error']}"

    news_list = news_data.get("news", [])

    if not news_list:
        return "No se encontraron noticias recientes en este momento."

    today = datetime.now().strftime("%A %d de %B de %Y")
    days_es = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }
    months_es = {
        "January": "Enero", "February": "Febrero", "March": "Marzo", "April": "Abril",
        "May": "Mayo", "June": "Junio", "July": "Julio", "August": "Agosto",
        "September": "Septiembre", "October": "Octubre", "November": "Noviembre", "December": "Diciembre"
    }

    for eng, esp in days_es.items():
        today = today.replace(eng, esp)
    for eng, esp in months_es.items():
        today = today.replace(eng, esp)

    # Listar las fuentes presentes
    fuentes = sorted(set(n["fuente"] for n in news_list))
    formatted_text = f"📰 Noticias de {', '.join(fuentes)}\n"
    formatted_text += f"📅 Fecha: {today}\n"
    formatted_text += f"Total de noticias: {news_data.get('total', 0)}\n\n"
    
    for idx, news in enumerate(news_list, 1):
        formatted_text += f"{idx}. [{news['fuente']}] {news['titulo']}\n"
        if news.get('descripcion'):
            # Limpiar HTML tags si existen
            desc = news['descripcion']
            if '<' in desc:
                desc = BeautifulSoup(desc, 'html.parser').get_text()
            # Limitar descripción si es muy larga
            if len(desc) > 200:
                desc = desc[:200] + "..."
            formatted_text += f"   {desc}\n"
        
        if news.get('fecha'):
            # Intentar formatear la fecha de manera más legible
            try:
                # Los RSS suelen usar formato RFC 2822
                date_obj = datetime.strptime(news['fecha'], '%a, %d %b %Y %H:%M:%S %z')
                formatted_date = date_obj.strftime("%d/%m/%Y %H:%M")
                formatted_text += f"   📅 {formatted_date}\n"
            except (ValueError, TypeError):
                formatted_text += f"   📅 {news['fecha']}\n"
        
        formatted_text += f"   🔗 {news['url']}\n\n"
    
    return formatted_text.strip()

def get_newspapers_by_source(source: str, limit: int = 5) -> Dict:
    """
    Obtiene noticias de un periódico específico.

    Args:
        source: Clave del periódico (ej: "elpais", "abc", "lavanguardia", etc.)
        limit: Número de noticias a obtener
    """
    source_key = source.lower()
    if source_key not in RSS_SOURCES:
        available = ", ".join(RSS_SOURCES.keys())
        return {
            "error": f"Fuente no reconocida: {source}. Disponibles: {available}",
            "news": []
        }

    news = get_news_by_source(source_key, limit=limit)
    return {
        "total": len(news),
        "fecha_consulta": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "news": news
    }
