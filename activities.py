from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import unicodedata

load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"
SEARCH_QUERY_MODEL = "gpt-5.4-mini"
RESULT_PERSONALIZATION_MODEL = "gpt-5.4-nano"

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

INTEREST_QUERY_RULES = [
    (
        ("ajedrez", "escacs"),
        {
            "summary": "He buscado clubes de ajedrez y espacios tranquilos para jugar partidas y socializar.",
            "queries": [
                "club de ajedrez",
                "asociacion de ajedrez",
                "centro civico ajedrez",
            ],
        },
    ),
    (
        ("lectura", "libros", "leer"),
        {
            "summary": "He buscado bibliotecas y espacios tranquilos con actividades de lectura.",
            "queries": [
                "biblioteca municipal",
                "club de lectura",
                "centro cultural lectura",
            ],
        },
    ),
    (
        ("museo", "arte", "pintura"),
        {
            "summary": "He buscado museos y centros culturales con planes tranquilos e interesantes.",
            "queries": [
                "museo",
                "centro cultural",
                "sala de exposiciones",
            ],
        },
    ),
    (
        ("musica", "cantar", "coro"),
        {
            "summary": "He buscado actividades musicales y espacios donde disfrutar de la musica con calma.",
            "queries": [
                "escuela de musica",
                "centro civico musica",
                "coro local",
            ],
        },
    ),
]


def _extract_json_content(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _keyword_fallback_search_plan(user_profile: dict) -> dict:
    interests = user_profile.get("interests", "") or ""
    description = user_profile.get("description", "") or ""
    profile_text = _normalize_text(f"{interests} {description}")

    for keywords, plan in INTEREST_QUERY_RULES:
        if any(keyword in profile_text for keyword in keywords):
            return {
                "summary": plan["summary"],
                "queries": plan["queries"][:5],
            }

    return {
        "summary": "He buscado actividades tranquilas y sociales que encajen con el perfil del usuario.",
        "queries": ["centro de mayores", "biblioteca", "centro cívico"],
    }


def _build_search_summary_from_queries(queries: list[str]) -> str:
    if not queries:
        return "He buscado actividades interesantes cerca de ti."
    preview = ", ".join(queries[:2])
    return f"He buscado opciones como {preview} cerca de ti."


def get_search_plan(user_profile: dict, tutor_factors: str = "") -> dict:
    """
    Build a brief search summary and 3-5 Google Places queries from the user's profile.
    """
    interests = user_profile.get("interests", "") or ""
    description = user_profile.get("description", "") or ""
    city = user_profile.get("city", "") or ""

    prompt = (
        "Eres un asistente que prepara busquedas en Google Places para personas mayores.\n"
        f"Perfil del usuario:\n"
        f"- Intereses: {interests}\n"
        f"- Descripcion: {description}\n"
        f"- Ciudad: {city}\n"
    )
    if tutor_factors:
        prompt += f"- Condiciones de salud/factores importantes: {tutor_factors}\n"
    prompt += (
        "\nPiensa en actividades o lugares MUY concretos que encajen con sus intereses. "
        "Si le gusta el ajedrez, busca cosas como clubes o asociaciones de ajedrez. "
        "Si le gusta leer, piensa en bibliotecas o clubes de lectura. "
        "Evita terminos genericos si hay una aficion clara.\n"
        "Responde SOLO con un JSON object con este formato:\n"
        '{"summary":"Breve resumen de la busqueda","queries":["query 1","query 2","query 3"]}\n'
        "Las queries deben ser utiles para Google Places y tener entre 2 y 5 elementos."
    )

    llm_output = _llm_json(
        prompt,
        temperature=0.2,
        max_tokens=220,
        model=SEARCH_QUERY_MODEL,
    )

    if isinstance(llm_output, dict):
        queries = llm_output.get("queries", [])
        summary = llm_output.get("summary", "")
        cleaned_queries = [
            query.strip()
            for query in queries
            if isinstance(query, str) and query.strip()
        ]
        if cleaned_queries:
            return {
                "summary": summary.strip() or _build_search_summary_from_queries(cleaned_queries),
                "queries": cleaned_queries[:5],
            }

    if isinstance(llm_output, list):
        cleaned_queries = [
            query.strip()
            for query in llm_output
            if isinstance(query, str) and query.strip()
        ]
        if cleaned_queries:
            fallback_plan = _keyword_fallback_search_plan(user_profile)
            return {
                "summary": fallback_plan["summary"] if any(
                    keyword in _normalize_text(" ".join(cleaned_queries))
                    for keywords, _ in INTEREST_QUERY_RULES
                    for keyword in keywords
                ) else _build_search_summary_from_queries(cleaned_queries),
                "queries": cleaned_queries[:5],
            }

    return _keyword_fallback_search_plan(user_profile)


def _llm_json(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 250,
    model: str = RESULT_PERSONALIZATION_MODEL,
):
    if not openai_client:
        return None

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        return json.loads(_extract_json_content(content))
    except Exception:
        return None


def geocode_city(city: str) -> dict:
    """
    Convert a city name to lat/lng using Google Geocoding API.

    Returns:
        {"lat": float, "lng": float} or {"error": str}
    """
    if not GOOGLE_PLACES_API_KEY:
        return {"error": "No se ha configurado GOOGLE_PLACES_API_KEY."}

    try:
        response = requests.get(
            GEOCODING_URL,
            params={
                "address": f"{city}, Spain",
                "key": GOOGLE_PLACES_API_KEY,
                "language": "es",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            return {"error": f"No se encontró la ciudad '{city}'."}
        location = results[0]["geometry"]["location"]
        return {"lat": location["lat"], "lng": location["lng"]}
    except requests.exceptions.RequestException as exc:
        return {"error": f"Error al geocodificar la ciudad: {str(exc)}"}


def get_search_queries(user_profile: dict, tutor_factors: str = "") -> list[str]:
    """
    Use a small LLM plus profile-based fallback to generate Google Places queries.
    """
    return get_search_plan(user_profile, tutor_factors).get("queries", [])


def search_places(queries: list[str], lat: float, lng: float, radius_m: int = 10000) -> list[dict]:
    """
    Search Google Places Nearby for each query and deduplicate by place_id.
    """
    if not GOOGLE_PLACES_API_KEY:
        return []

    seen_ids = set()
    places = []

    for query in queries:
        try:
            response = requests.get(
                PLACES_NEARBY_URL,
                params={
                    "location": f"{lat},{lng}",
                    "radius": radius_m,
                    "keyword": query,
                    "language": "es",
                    "key": GOOGLE_PLACES_API_KEY,
                },
                timeout=10,
            )
            response.raise_for_status()
            results = response.json().get("results", [])

            for place in results:
                place_id = place.get("place_id")
                if not place_id or place_id in seen_ids:
                    continue

                seen_ids.add(place_id)
                places.append(
                    {
                        "name": place.get("name", ""),
                        "address": place.get("vicinity", ""),
                        "rating": place.get("rating"),
                        "open_now": place.get("opening_hours", {}).get("open_now"),
                        "types": place.get("types", []),
                        "place_id": place_id,
                    }
                )
        except requests.exceptions.RequestException:
            continue

    return places


def personalize_results(places: list[dict], user_profile: dict, tutor_factors: str = "") -> list[dict]:
    """
    Use gpt-5.4-nano to rank places and add a warm recommendation.
    """
    if not places:
        return []

    interests = user_profile.get("interests", "") or ""
    description = user_profile.get("description", "") or ""

    places_text = "\n".join(
        (
            f"- {place['name']} ({place['address']}) - rating: {place.get('rating', 'N/A')}, "
            f"abierto: {'si' if place.get('open_now') else 'no' if place.get('open_now') is False else 'desconocido'}, "
            f"tipos: {', '.join(place.get('types', [])[:3])}"
        )
        for place in places[:20]
    )

    prompt = (
        "Eres un asistente calido que ayuda a personas mayores.\n"
        f"Perfil del usuario:\n"
        f"- Intereses: {interests}\n"
        f"- Descripcion: {description}\n"
    )
    if tutor_factors:
        prompt += f"- Condiciones de salud: {tutor_factors}\n"
    prompt += (
        f"\nLugares encontrados cerca del usuario:\n{places_text}\n\n"
        "Selecciona los 3 a 5 lugares mas apropiados para esta persona. "
        "Para cada uno, escribe una frase corta y calida explicando por que le puede interesar. "
        "Responde SOLO con un JSON array de objetos con los campos "
        '"name" y "recommendation". Ejemplo: '
        '[{"name": "Biblioteca Municipal", "recommendation": "Tienen un club de lectura los martes."}]'
    )

    recommendations = _llm_json(
        prompt,
        temperature=0.5,
        max_tokens=500,
        model=RESULT_PERSONALIZATION_MODEL,
    )
    if isinstance(recommendations, list):
        recommendation_map = {}
        for recommendation in recommendations:
            if (
                isinstance(recommendation, dict)
                and isinstance(recommendation.get("name"), str)
                and isinstance(recommendation.get("recommendation"), str)
            ):
                recommendation_map[recommendation["name"]] = recommendation["recommendation"].strip()

        personalized = []
        for place in places:
            if place["name"] in recommendation_map:
                personalized.append(
                    {
                        **place,
                        "recommendation": recommendation_map[place["name"]],
                    }
                )
        if personalized:
            return personalized[:5]

    return [
        {**place, "recommendation": "Un lugar interesante cerca de ti."}
        for place in places[:5]
    ]


def format_activities_for_chat(activities: list[dict], search_summary: str = "") -> str:
    """Format the personalized activities list for the chatbot."""
    if not activities:
        return "No se encontraron actividades cercanas."

    lines = [search_summary.strip(), ""] if search_summary.strip() else []
    for index, activity in enumerate(activities, 1):
        rating_text = f"⭐ {activity['rating']}" if activity.get("rating") else ""
        open_text = (
            "Abierto ahora"
            if activity.get("open_now")
            else "Cerrado ahora"
            if activity.get("open_now") is False
            else ""
        )
        status = " | ".join(part for part in [rating_text, open_text] if part)

        lines.append(f"{index}. {activity['name']} - {activity['address']}")
        if status:
            lines.append(f"   {status}")
        lines.append(f"   -> {activity.get('recommendation', '')}")
        lines.append("")

    return "\n".join(lines).strip()


def search_activities(
    user_profile: dict,
    tutor_factors: str = "",
    latitude: float = None,
    longitude: float = None,
    radius_km: int = 10,
) -> str:
    """
    Search for personalized activities near the user.
    """
    if not GOOGLE_PLACES_API_KEY:
        return "Error: no se ha configurado la clave de Google Places."

    lat = latitude
    lng = longitude

    if lat is None or lng is None:
        city = (user_profile or {}).get("city", "")
        if not city:
            return (
                "No conozco tu ubicación. Puedes decirme tu ciudad "
                "o activar la ubicación en tu navegador."
            )

        geocoded = geocode_city(city)
        if "error" in geocoded:
            return geocoded["error"]
        lat = geocoded["lat"]
        lng = geocoded["lng"]

    search_plan = get_search_plan(user_profile or {}, tutor_factors)
    queries = search_plan.get("queries", [])
    places = search_places(queries, lat, lng, radius_m=radius_km * 1000)

    if not places:
        return (
            "No he encontrado actividades cercanas con ese criterio. "
            "¿Quieres que busque en un radio más amplio?"
        )

    personalized = personalize_results(places, user_profile or {}, tutor_factors)
    return format_activities_for_chat(
        personalized,
        search_summary=search_plan.get("summary", ""),
    )
