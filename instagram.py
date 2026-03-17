import re
import requests
from bs4 import BeautifulSoup
import json


INSTAGRAM_USERNAME_REGEX = re.compile(r'^[a-zA-Z0-9._]{1,30}$')

INSTAGRAM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


def validate_instagram_username(username: str) -> dict:
    """
    Validates an Instagram username format and returns structured profile info.

    Args:
        username: Instagram username or URL

    Returns:
        Dict with validated username and profile URL, or error
    """
    # Extract username from URL if provided
    cleaned = extract_username(username)

    if not cleaned:
        return {"error": "No se pudo extraer un nombre de usuario válido de Instagram."}

    if not INSTAGRAM_USERNAME_REGEX.match(cleaned):
        return {
            "error": (
                f"El nombre de usuario '{cleaned}' no es válido. "
                "Solo puede contener letras, números, puntos y guiones bajos (máx. 30 caracteres)."
            )
        }

    return {
        "username": cleaned,
        "profile_url": f"https://www.instagram.com/{cleaned}/",
    }


def extract_username(input_str: str) -> str | None:
    """
    Extracts an Instagram username from various input formats:
    - Plain username: 'johndoe'
    - With @: '@johndoe'
    - Full URL: 'https://www.instagram.com/johndoe/'
    - Short URL: 'instagram.com/johndoe'
    """
    input_str = input_str.strip()

    if not input_str:
        return None

    # Handle URLs
    url_pattern = re.compile(
        r'(?:https?://)?(?:www\.)?instagram\.com/([a-zA-Z0-9._]+)/?(?:\?.*)?$'
    )
    match = url_pattern.match(input_str)
    if match:
        return match.group(1)

    # Handle @username
    if input_str.startswith('@'):
        input_str = input_str[1:]

    # Return as plain username (will be validated by caller)
    return input_str if input_str else None


def fetch_instagram_profile(username: str) -> dict | None:
    """
    Attempts to check if an Instagram profile exists by checking the public page.
    Returns basic info if accessible, None if not reachable.

    Note: This is a best-effort check. Instagram may rate-limit or block requests.
    """
    try:
        url = f"https://www.instagram.com/{username}/"
        response = requests.get(
            url,
            headers=INSTAGRAM_HEADERS,
            timeout=5,
            allow_redirects=True,
        )
        if response.status_code == 200:
            return {"exists": True, "username": username, "profile_url": url}
        return {"exists": False, "username": username, "profile_url": url}
    except requests.RequestException:
        return None


def get_instagram_info(username: str) -> dict:
    """
    Fetches public Instagram profile information by scraping the profile page.
    Extracts data from HTML meta tags (og:description, title, etc.).

    Args:
        username: Instagram username (without @)

    Returns:
        Dict with profile info: name, bio, followers, following, posts, profile_url
        or dict with error key if failed
    """
    validation = validate_instagram_username(username)
    if validation.get("error"):
        return validation

    clean_username = validation["username"]
    url = f"https://www.instagram.com/{clean_username}/"

    try:
        response = requests.get(url, headers=INSTAGRAM_HEADERS, timeout=8)

        if response.status_code == 404:
            return {"error": f"El perfil @{clean_username} no existe en Instagram."}
        if response.status_code != 200:
            return {"error": f"No se pudo acceder al perfil de Instagram (código {response.status_code})."}

        soup = BeautifulSoup(response.text, "lxml")
        info = _parse_profile_from_html(soup, clean_username)
        info["profile_url"] = url
        return info

    except requests.Timeout:
        return {"error": "Instagram tardó demasiado en responder."}
    except requests.RequestException as e:
        return {"error": f"Error de conexión con Instagram: {str(e)}"}


def _parse_profile_from_html(soup: BeautifulSoup, username: str) -> dict:
    """
    Extracts profile info from Instagram HTML meta tags.

    og:description typically contains:
      "1,234 Followers, 567 Following, 89 Posts - See Instagram photos and videos from Display Name (@username)"

    og:title typically contains:
      "Display Name (@username) • Instagram photos and videos"
    """
    info = {"username": username, "nombre": None, "biografia": None,
            "seguidores": None, "seguidos": None, "publicaciones": None}

    # Extract from og:description
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        desc = og_desc["content"]
        info.update(_parse_og_description(desc))

    # Extract display name from og:title
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"]
        # Format: "Display Name (@username) • Instagram photos and videos"
        name_match = re.match(r'^(.+?)\s*\(@', title)
        if name_match:
            info["nombre"] = name_match.group(1).strip()

    # Try description meta tag for bio
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        desc = meta_desc["content"]
        if not info.get("biografia"):
            info.update(_parse_og_description(desc))

    # Try to extract from JSON-LD structured data
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            ld_data = json.loads(script.string)
            if isinstance(ld_data, dict):
                if ld_data.get("@type") == "ProfilePage":
                    main = ld_data.get("mainEntity", {})
                    if main.get("alternateName"):
                        info["username"] = main["alternateName"].lstrip("@")
                    if main.get("name") and not info["nombre"]:
                        info["nombre"] = main["name"]
                    if main.get("description") and not info["biografia"]:
                        info["biografia"] = main["description"]
                    stats = main.get("interactionStatistic", [])
                    for stat in stats:
                        stat_type = stat.get("interactionType", "")
                        count = stat.get("userInteractionCount")
                        if "Follow" in stat_type and count is not None:
                            info["seguidores"] = _format_count(count)
                        if stat.get("name") == "Follows" and count is not None:
                            info["seguidos"] = _format_count(count)
                    if main.get("agentInteractionStatistic"):
                        agent_stat = main["agentInteractionStatistic"]
                        if agent_stat.get("userInteractionCount") is not None:
                            info["publicaciones"] = _format_count(agent_stat["userInteractionCount"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    return info


def _parse_og_description(desc: str) -> dict:
    """
    Parses Instagram og:description format:
    '1,234 Followers, 567 Following, 89 Posts - See Instagram photos and videos from Name (@user)'

    Also handles Spanish locale:
    '1234 seguidores, 567 seguidos, 89 publicaciones...'
    """
    result = {}

    # English format
    followers_match = re.search(r'([\d,.KkMm]+)\s*Followers', desc, re.IGNORECASE)
    following_match = re.search(r'([\d,.KkMm]+)\s*Following', desc, re.IGNORECASE)
    posts_match = re.search(r'([\d,.KkMm]+)\s*Posts', desc, re.IGNORECASE)

    # Spanish format
    if not followers_match:
        followers_match = re.search(r'([\d,.KkMm]+)\s*seguidores', desc, re.IGNORECASE)
    if not following_match:
        following_match = re.search(r'([\d,.KkMm]+)\s*seguidos', desc, re.IGNORECASE)
    if not posts_match:
        posts_match = re.search(r'([\d,.KkMm]+)\s*publicaciones', desc, re.IGNORECASE)

    if followers_match:
        result["seguidores"] = followers_match.group(1).strip()
    if following_match:
        result["seguidos"] = following_match.group(1).strip()
    if posts_match:
        result["publicaciones"] = posts_match.group(1).strip()

    # Extract bio — text after the dash separator
    bio_match = re.search(r'Posts?\s*[-–—]\s*(.+)', desc, re.IGNORECASE)
    if not bio_match:
        bio_match = re.search(r'publicaciones\s*[-–—]\s*(.+)', desc, re.IGNORECASE)
    if bio_match:
        bio_text = bio_match.group(1).strip()
        # Remove the "See Instagram photos..." prefix
        bio_text = re.sub(r'^See Instagram photos and videos from\s*', '', bio_text, flags=re.IGNORECASE)
        # Remove trailing "(@username)" part
        bio_text = re.sub(r'\s*\(@[\w.]+\)\s*$', '', bio_text)
        if bio_text:
            result["biografia"] = bio_text

    return result


def _format_count(count) -> str:
    """Formats a numeric count to a readable string."""
    try:
        n = int(count)
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 10_000:
            return f"{n / 1_000:.1f}K"
        return f"{n:,}".replace(",", ".")
    except (ValueError, TypeError):
        return str(count)


def format_instagram_for_chat(info: dict) -> str:
    """
    Formats Instagram profile info into a readable string for the chatbot.
    """
    if info.get("error"):
        return f"Error al obtener información de Instagram: {info['error']}"

    parts = [f"Información de Instagram de @{info.get('username', '?')}:"]

    if info.get("nombre"):
        parts.append(f"- Nombre: {info['nombre']}")
    if info.get("biografia"):
        parts.append(f"- Biografía: {info['biografia']}")
    if info.get("seguidores"):
        parts.append(f"- Seguidores: {info['seguidores']}")
    if info.get("seguidos"):
        parts.append(f"- Seguidos: {info['seguidos']}")
    if info.get("publicaciones"):
        parts.append(f"- Publicaciones: {info['publicaciones']}")
    if info.get("profile_url"):
        parts.append(f"- Perfil: {info['profile_url']}")

    # If we got almost nothing useful, note it
    if len(parts) <= 2:
        parts.append("- No se pudo obtener información detallada. El perfil puede ser privado.")

    return "\n".join(parts)
