"""
CineMatch AI - Next-Gen Intelligent Movie Recommender & Analyzer
Features:
- Content-Based Recommendation with Vector Similarity
- Explainable AI (XAI) Thematic Keyword Breakdown
- Direct YouTube Official Trailer Integration
- Interactive Movie vs. Movie Match Analyzer
- Session Watchlist Bookmarking
- Ultra-Fast In-Memory Caching & Resilient Offline Fallback
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st

# Ensure local imports work reliably
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from utils import (
    find_data_file,
    fetch_movie_details,
    generate_svg_poster,
    extract_shared_keywords,
    get_trailer_search_url,
    interpret_match_tier,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (Cinema Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch AI | Next-Gen Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 50%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .badge-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.4);
        margin-right: 0.5rem;
    }

    /* Movie Poster Card */
    .movie-card {
        background: #0f172a;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        margin-bottom: 0.8rem;
    }
    .movie-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.6);
    }
    .poster-img {
        width: 100%;
        height: 320px;
        object-fit: cover;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background-color: #1e293b;
    }
    .movie-info {
        padding: 0.9rem;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-title {
        font-weight: 700;
        font-size: 0.95rem;
        line-height: 1.25;
        color: #f8fafc;
        margin-bottom: 0.4rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 2.4rem;
    }
    .movie-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.3rem;
        font-size: 0.8rem;
    }
    .rating-badge {
        color: #facc15;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.2rem;
    }
    .similarity-pill {
        background: #4338ca;
        color: #e0e7ff;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
    }
    .year-text {
        color: #94a3b8;
    }
    .tag-chip {
        display: inline-block;
        background: rgba(148, 163, 184, 0.15);
        color: #cbd5e1;
        font-size: 0.7rem;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        margin: 0.15rem 0.15rem 0 0;
    }

    /* Comparison Box */
    .compare-container {
        background: #1e293b;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        text-align: center;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        border-color: #6366f1;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Cached Resource & Data Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="🎬 Loading CineMatch AI Models & Vector Space...")
def load_recommender_models() -> Tuple[pd.DataFrame, Any]:
    """Loads movie metadata dictionary and precomputed similarity matrix."""
    dict_file = find_data_file("movie_dict.pkl")
    sim_file = find_data_file("similarity.pkl")

    with open(dict_file, "rb") as f:
        movies_dict = pickle.load(f)
    movies_df = pd.DataFrame(movies_dict)

    with open(sim_file, "rb") as f:
        similarity_matrix = pickle.load(f)

    return movies_df, similarity_matrix


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_movie_details(movie_id: int, title: str, api_key: str) -> Dict[str, Any]:
    """Caches TMDB API results per movie to eliminate network latency."""
    return fetch_movie_details(movie_id=movie_id, movie_title=title, api_key=api_key)


# Initialize Session State
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# Load models
try:
    movies, similarity = load_recommender_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Error loading recommender models: {e}")
    st.info("Ensure `movie_dict.pkl` and `similarity.pkl` are located in your project directory.")


# -----------------------------------------------------------------------------
# 3. Recommendation Engine Logic
# -----------------------------------------------------------------------------
def get_recommendations(
    movie_title: str,
    top_n: int = 5,
    api_key: str = "",
) -> List[Dict[str, Any]]:
    """Returns top N recommendations with similarity scores, XAI keywords, and TMDB metadata."""
    if movie_title not in movies["title"].values:
        return []

    target_idx = movies[movies["title"] == movie_title].index[0]
    target_tags = movies.iloc[target_idx]["tags"]
    distances = similarity[target_idx]

    ranked = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : top_n + 1]

    recommendation_targets = []
    for idx, sim_score in ranked:
        row = movies.iloc[idx]
        shared_kw = extract_shared_keywords(target_tags, row["tags"], top_k=4)
        recommendation_targets.append((int(row.movie_id), str(row.title), float(sim_score), shared_kw))

    # Fetch TMDB metadata in parallel
    results = []
    with ThreadPoolExecutor(max_workers=min(top_n, 8)) as executor:
        futures = [
            (
                executor.submit(get_cached_movie_details, m_id, m_title, api_key),
                score,
                kw,
            )
            for m_id, m_title, score, kw in recommendation_targets
        ]
        for future, score, kw in futures:
            try:
                details = future.result()
            except Exception:
                details = {
                    "movie_id": 0,
                    "title": "Movie",
                    "poster_url": generate_svg_poster("Movie"),
                    "rating": None,
                    "release_year": "N/A",
                    "genres": [],
                    "overview": "Overview unavailable.",
                    "tmdb_url": "#",
                    "trailer_url": "#",
                }
            details["match_percentage"] = int(round(score * 100))
            details["similarity_raw"] = score
            details["shared_keywords"] = kw
            results.append(details)

    return results


# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    num_recommendations = st.slider(
        "Recommendations Count",
        min_value=4,
        max_value=10,
        value=5,
        step=1,
    )

    sort_by = st.selectbox(
        "Sort Results By",
        options=["Vector Similarity", "TMDB Rating", "Release Year"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🔑 TMDB API Key (Optional)")
    user_api_key = st.text_input(
        "Custom API Key",
        value=os.getenv("TMDB_API_KEY", ""),
        type="password",
        help="Automatically loaded from .env if present. Leave blank to use the built-in demo key.",
    )

    st.markdown("---")
    st.markdown("### 📌 Session Watchlist")
    if st.session_state.watchlist:
        st.success(f"{len(st.session_state.watchlist)} movie(s) saved")
        for w_item in st.session_state.watchlist[-5:]:
            st.caption(f"• {w_item['title']} ({w_item.get('release_year', 'N/A')})")
        if st.button("🗑️ Clear Watchlist", use_container_width=True):
            st.session_state.watchlist = []
            st.rerun()
    else:
        st.caption("Your watchlist is currently empty.")

    st.markdown("---")
    st.markdown("### 📊 Telemetry")
    if model_loaded:
        st.metric(label="Total Movies Indexed", value=f"{len(movies):,}")
        st.metric(label="Similarity Matrix", value=f"{similarity.shape[0]} × {similarity.shape[1]}")
    st.caption("Cosine Similarity on TMDB 5000 Tag Vectors.")


# -----------------------------------------------------------------------------
# 5. Main UI & Tabs
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎬 CineMatch AI</div>
        <div class="hero-subtitle">Next-Generation Content-Based Movie Recommendation & Intelligence Engine</div>
        <div>
            <span class="badge-pill">Natural Language Processing</span>
            <span class="badge-pill">Explainable AI (XAI)</span>
            <span class="badge-pill">Cosine Similarity</span>
            <span class="badge-pill">TMDB Integration</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_loaded:
    st.stop()

all_titles = movies["title"].values.tolist()

tab_recommend, tab_compare, tab_watchlist = st.tabs([
    "🎯 Discover Recommendations",
    "⚖️ Movie Match Analyzer (Compare 2)",
    f"📌 Watchlist ({len(st.session_state.watchlist)})",
])

# -----------------------------------------------------------------------------
# TAB 1: Discover Recommendations
# -----------------------------------------------------------------------------
with tab_recommend:
    col_search, col_surprise = st.columns([3.5, 1.2])

    with col_search:
        if "selected_movie" not in st.session_state:
            st.session_state.selected_movie = all_titles[0]

        selected_movie_name = st.selectbox(
            "🔍 Search or select a movie you like:",
            options=all_titles,
            index=all_titles.index(st.session_state.selected_movie)
            if st.session_state.selected_movie in all_titles
            else 0,
            key="movie_selector",
        )

    with col_surprise:
        st.write("")
        st.write("")
        if st.button("🎲 Surprise Me!", use_container_width=True):
            st.session_state.selected_movie = random.choice(all_titles)
            st.rerun()

    st.session_state.selected_movie = selected_movie_name

    # Selected Movie Profile Card
    curr_row = movies[movies["title"] == selected_movie_name].iloc[0]
    curr_details = get_cached_movie_details(int(curr_row.movie_id), selected_movie_name, user_api_key)

    with st.expander("📌 Selected Movie Profile & Metadata", expanded=False):
        sel_col1, sel_col2 = st.columns([1, 4])
        with sel_col1:
            st.image(curr_details["poster_url"], width=180)
        with sel_col2:
            st.subheader(f"{curr_details['title']} ({curr_details['release_year']})")
            if curr_details.get("rating"):
                st.markdown(f"⭐ **TMDB Rating:** `{curr_details['rating']} / 10`")
            if curr_details.get("genres"):
                st.markdown(f"🏷️ **Genres:** {', '.join(curr_details['genres'])}")
            st.markdown(f"**Overview:** {curr_details['overview']}")
            st.markdown(f"[▶️ Watch Official Trailer on YouTube ↗]({curr_details['trailer_url']})")

    st.write("")
    recommend_clicked = st.button("✨ Discover Similar Movies", type="primary", use_container_width=True)

    if recommend_clicked or "has_recommended" in st.session_state:
        st.session_state.has_recommended = True

        with st.spinner("Analyzing semantic vectors and retrieving posters..."):
            recs = get_recommendations(
                movie_title=selected_movie_name,
                top_n=num_recommendations,
                api_key=user_api_key,
            )

        if sort_by == "TMDB Rating":
            recs = sorted(recs, key=lambda x: (x.get("rating") is not None, x.get("rating") or 0), reverse=True)
        elif sort_by == "Release Year":
            recs = sorted(recs, key=lambda x: str(x.get("release_year") or ""), reverse=True)

        st.markdown(f"### 🎯 Top Recommendations for *{selected_movie_name}*")
        st.caption(f"Sorted by {sort_by} • Showing {len(recs)} matches")

        cols_per_row = 5
        for row_start in range(0, len(recs), cols_per_row):
            row_items = recs[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_items))

            for idx, item in enumerate(row_items):
                with cols[idx]:
                    rating_str = f"⭐ {item['rating']}" if item.get("rating") else "⭐ N/A"
                    year_str = str(item.get("release_year", "N/A"))
                    match_str = f"{item.get('match_percentage', 0)}% Match"

                    card_html = f"""
                    <div class="movie-card">
                        <img class="poster-img" src="{item['poster_url']}" alt="{item['title']}" loading="lazy"/>
                        <div class="movie-info">
                            <div class="movie-title" title="{item['title']}">{item['title']}</div>
                            <div class="movie-meta">
                                <span class="rating-badge">{rating_str}</span>
                                <span class="year-text">{year_str}</span>
                            </div>
                            <div style="margin-top: 0.5rem; text-align: right;">
                                <span class="similarity-pill">{match_str}</span>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

                    # Explainable AI: Shared Themes
                    if item.get("shared_keywords"):
                        tags_html = " ".join([f'<span class="tag-chip">#{kw}</span>' for kw in item["shared_keywords"]])
                        st.markdown(f"<div style='margin-bottom: 0.5rem;'>{tags_html}</div>", unsafe_allow_html=True)

                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        st.markdown(f"<a href='{item['trailer_url']}' target='_blank' style='text-decoration:none;'><button style='width:100%; border-radius:6px; background:#4338ca; color:#fff; border:none; padding:4px 0; font-size:12px; cursor:pointer;'>▶ Trailer</button></a>", unsafe_allow_html=True)
                    with btn_c2:
                        # Watchlist bookmark button
                        already_saved = any(w["title"] == item["title"] for w in st.session_state.watchlist)
                        btn_label = "Saved ✓" if already_saved else "➕ Save"
                        if st.button(btn_label, key=f"wl_{item['title']}", use_container_width=True):
                            if not already_saved:
                                st.session_state.watchlist.append(item)
                                st.rerun()

                    with st.expander("Synopsis"):
                        if item.get("genres"):
                            st.caption(f"**Genres:** {', '.join(item['genres'])}")
                        st.write(item.get("overview", "No synopsis available."))
                        if item.get("tmdb_url") and item["tmdb_url"] != "#":
                            st.markdown(f"[View on TMDB ↗]({item['tmdb_url']})")


# -----------------------------------------------------------------------------
# TAB 2: Movie Match Analyzer (Compare 2 Movies)
# -----------------------------------------------------------------------------
with tab_compare:
    st.subheader("⚖️ Movie vs. Movie Thematic Affinity Analyzer")
    st.caption("Compare any two films in the vector space to analyze similarity percentage, affinity tier, and common elements.")

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        movie_a_name = st.selectbox("Select Film A:", options=all_titles, index=0, key="cmp_a")
    with c_col2:
        movie_b_name = st.selectbox("Select Film B:", options=all_titles, index=min(1, len(all_titles) - 1), key="cmp_b")

    if movie_a_name and movie_b_name:
        idx_a = movies[movies["title"] == movie_a_name].index[0]
        idx_b = movies[movies["title"] == movie_b_name].index[0]

        row_a = movies.iloc[idx_a]
        row_b = movies.iloc[idx_b]

        sim_val = float(similarity[idx_a][idx_b])
        pct_match = round(sim_val * 100, 1)
        tier_info = interpret_match_tier(pct_match)

        common_kw = extract_shared_keywords(row_a["tags"], row_b["tags"], top_k=6)

        details_a = get_cached_movie_details(int(row_a.movie_id), movie_a_name, user_api_key)
        details_b = get_cached_movie_details(int(row_b.movie_id), movie_b_name, user_api_key)

        st.markdown(
            f"""
            <div class="compare-container">
                <div style="font-size: 1.1rem; color: #94a3b8; margin-bottom: 0.3rem;">THEMATIC AFFINITY SCORE</div>
                <div style="font-size: 3rem; font-weight: 800; color: {tier_info['color']};">
                    {pct_match}% {tier_info['icon']}
                </div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-top: 0.2rem;">
                    {tier_info['tier']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.image(details_a["poster_url"], width=220)
            st.markdown(f"### {details_a['title']} ({details_a['release_year']})")
            if details_a.get("rating"):
                st.markdown(f"⭐ **Rating:** `{details_a['rating']} / 10`")
            st.write(details_a.get("overview", ""))

        with cmp_col2:
            st.image(details_b["poster_url"], width=220)
            st.markdown(f"### {details_b['title']} ({details_b['release_year']})")
            if details_b.get("rating"):
                st.markdown(f"⭐ **Rating:** `{details_b['rating']} / 10`")
            st.write(details_b.get("overview", ""))

        if common_kw:
            st.markdown("#### 🔗 Shared Thematic Vectors")
            chips = " ".join([f'<span class="tag-chip" style="font-size:0.85rem; padding:0.3rem 0.6rem;">#{kw}</span>' for kw in common_kw])
            st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 3: Session Watchlist
# -----------------------------------------------------------------------------
with tab_watchlist:
    st.subheader("📌 Your Session Watchlist")
    st.caption("Movies saved during your current exploration session.")

    if not st.session_state.watchlist:
        st.info("No movies saved yet. Click **➕ Save** on any movie card in the Recommendations tab!")
    else:
        w_cols = st.columns(min(len(st.session_state.watchlist), 5))
        for w_idx, w_item in enumerate(st.session_state.watchlist):
            with w_cols[w_idx % 5]:
                st.image(w_item["poster_url"], width=180)
                st.markdown(f"**{w_item['title']}**")
                st.caption(f"⭐ {w_item.get('rating', 'N/A')} • {w_item.get('release_year', 'N/A')}")
                st.markdown(f"[▶️ Trailer]({w_item.get('trailer_url', '#')})")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        CineMatch AI • Powered by Streamlit • Scikit-Learn Vector Space • TMDB API • NLTK
    </div>
    """,
    unsafe_allow_html=True,
)
