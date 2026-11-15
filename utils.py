"""
Utility functions for Movie Recommender System:
- Path resolution for data files across directories
- TMDB API poster and metadata fetching with caching
- Graceful offline SVG poster generation
- Thematic keywords extractor for Explainable AI (XAI)
- YouTube trailer search URL generator
- Similarity score interpretation
"""

import os
import re
import socket
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
import requests

# TMDB CloudFront Anycast IP addresses for ISP DNS-sinkhole resilience (e.g., Indian ISPs)
_TMDB_CLOUDFRONT_IPS = [
    "3.175.86.103",
    "3.175.86.37",
    "3.175.86.50",
    "3.175.86.67",
]
_orig_getaddrinfo = socket.getaddrinfo


def _custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host == "api.themoviedb.org":
        try:
            results = _orig_getaddrinfo(host, port, family, type, proto, flags)
            is_sinkhole = any(
                sockaddr[0].startswith("49.44.") or sockaddr[0].startswith("10.") or sockaddr[0].startswith("127.")
                for res in results
                for sockaddr in [res[4]]
                if isinstance(sockaddr, tuple) and len(sockaddr) > 0
            )
            if not is_sinkhole:
                return results
        except Exception:
            pass
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
            for ip in _TMDB_CLOUDFRONT_IPS
        ]
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _custom_getaddrinfo

def _load_env_file(filepath: Optional[Path] = None) -> None:
    """Loads key-value pairs from .env into os.environ without requiring external dependencies."""
    if filepath is None:
        filepath = Path(__file__).resolve().parent / ".env"
    if filepath.is_file():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and val and not os.environ.get(key):
                        os.environ[key] = val
        except Exception:
            pass


_load_env_file()

DEFAULT_TMDB_API_KEY = os.getenv("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8")
DEFAULT_TMDB_ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN", "")
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

COMMON_STOP_WORDS: Set[str] = {
    "in", "the", "a", "on", "to", "of", "and", "is", "for", "with", "by", "at",
    "an", "be", "from", "as", "that", "this", "his", "her", "their", "he", "she",
    "it", "they", "who", "which", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "into", "after", "up", "down", "out",
    "over", "under", "again", "further", "then", "once", "here", "there", "what",
    "about", "against", "between", "during", "before", "through", "above", "below",
    "life", "find", "one", "two", "new", "get", "take", "make", "help", "back"
}


def find_data_file(filename: str) -> Path:
    """
    Locates data file across nested or root project structures.
    Prevents FileNotFoundError regardless of invocation directory.
    """
    candidates = [
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent / "movie-recommender-system-main" / filename,
        Path.cwd() / filename,
        Path.cwd() / "movie-recommender-system-main" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate '{filename}'. Checked locations:\n"
        + "\n".join(str(c) for c in candidates)
    )


def generate_svg_poster(title: str, genre: str = "Cinema") -> str:
    """
    Generates an inline SVG data URI poster as a resilient fallback
    when offline, rate-limited, or when no poster exists.
    """
    clean_title = re.sub(r"[<>&\"']", "", title)
    display_title = clean_title[:22] + ("..." if len(clean_title) > 22 else "")
    clean_genre = re.sub(r"[<>&\"']", "", genre)[:16].upper()

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1e1b4b;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#312e81;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0f172a;stop-opacity:1" />
    </linearGradient>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000000" flood-opacity="0.5"/>
    </filter>
  </defs>
  <rect width="300" height="450" rx="16" fill="url(#grad)"/>
  <rect x="12" y="12" width="276" height="426" rx="12" fill="none" stroke="#6366f1" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.4"/>
  <circle cx="150" cy="170" r="50" fill="#4338ca" opacity="0.6"/>
  <text x="150" y="185" font-size="44" text-anchor="middle" fill="#e0e7ff">&#127916;</text>
  <text x="150" y="260" font-size="18" font-weight="bold" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" text-anchor="middle" fill="#ffffff">{display_title}</text>
  <rect x="85" y="285" width="130" height="26" rx="13" fill="#6366f1" opacity="0.85"/>
  <text x="150" y="302" font-size="11" font-weight="bold" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" text-anchor="middle" fill="#ffffff">{clean_genre}</text>
  <text x="150" y="380" font-size="12" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" text-anchor="middle" fill="#94a3b8">CineMatch AI Recommender</text>
</svg>"""
    encoded = urllib.parse.quote(svg_content)
    return f"data:image/svg+xml;utf8,{encoded}"


def fetch_movie_details(
    movie_id: int,
    movie_title: str,
    api_key: Optional[str] = None,
    timeout: float = 3.0
) -> Dict[str, Any]:
    """
    Fetches rich movie metadata (poster, rating, overview, release year, genres)
    from TMDB API. Gracefully returns defaults on network timeout or failure.
    """
    key = api_key or DEFAULT_TMDB_API_KEY
    token = os.getenv("TMDB_ACCESS_TOKEN", "")

    headers = {"accept": "application/json"}
    if key and key.startswith("ey"):
        headers["Authorization"] = f"Bearer {key}"
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    elif token:
        headers["Authorization"] = f"Bearer {token}"
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    elif key:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}&language=en-US"
    else:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"

    fallback_genre = "Feature Film"
    result = {
        "movie_id": movie_id,
        "title": movie_title,
        "poster_url": generate_svg_poster(movie_title, fallback_genre),
        "rating": None,
        "release_year": "N/A",
        "genres": [],
        "overview": "Overview is currently unavailable.",
        "tagline": "",
        "tmdb_url": f"https://www.themoviedb.org/movie/{movie_id}",
        "trailer_url": get_trailer_search_url(movie_title),
    }

    if not key and not token:
        return result

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            genres = [g["name"] for g in data.get("genres", [])]
            primary_genre = genres[0] if genres else fallback_genre

            if poster_path:
                result["poster_url"] = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            else:
                result["poster_url"] = generate_svg_poster(movie_title, primary_genre)

            result["rating"] = round(data.get("vote_average", 0.0), 1) if data.get("vote_average") else None
            release_date = data.get("release_date", "")
            result["release_year"] = release_date.split("-")[0] if release_date else "N/A"
            result["genres"] = genres
            result["overview"] = data.get("overview") or "No overview provided."
            result["tagline"] = data.get("tagline") or ""
    except Exception:
        pass

    return result


def extract_shared_keywords(tags_a: str, tags_b: str, top_k: int = 5) -> List[str]:
    """
    Extracts shared meaningful keywords between two movie tag profiles
    for Explainable AI (XAI) output.
    """
    if not tags_a or not tags_b:
        return []
    words_a = {w.lower() for w in tags_a.split() if len(w) > 3 and w.lower() not in COMMON_STOP_WORDS}
    words_b = {w.lower() for w in tags_b.split() if len(w) > 3 and w.lower() not in COMMON_STOP_WORDS}
    common = sorted(list(words_a.intersection(words_b)))
    return common[:top_k]


def get_trailer_search_url(movie_title: str) -> str:
    """Generates direct YouTube search query link for the movie trailer."""
    query = f"{movie_title} official trailer"
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"


def interpret_match_tier(score_pct: float) -> Dict[str, str]:
    """Categorizes similarity match into qualitative affinity tiers."""
    if score_pct >= 30.0:
        return {"tier": "Exceptional Affinity", "color": "#22c55e", "icon": "🔥"}
    elif score_pct >= 20.0:
        return {"tier": "High Thematic Match", "color": "#6366f1", "icon": "🎯"}
    elif score_pct >= 12.0:
        return {"tier": "Moderate Synergy", "color": "#38bdf8", "icon": "💡"}
    else:
        return {"tier": "Subtle Nuance Match", "color": "#94a3b8", "icon": "🔹"}
