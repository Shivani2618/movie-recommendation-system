"""
End-to-End Machine Learning Pipeline for Movie Recommender System.
Processes TMDB 5000 Movie & Credits datasets, builds feature tags,
computes Cosine Similarity matrix, and exports models to pickle.

Usage:
    python train_model.py --movies tmdb_5000_movies.csv --credits tmdb_5000_credits.csv
"""

import ast
import argparse
from pathlib import Path
import pickle
from typing import Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    import nltk
    from nltk.stem.porter import PorterStemmer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    PorterStemmer = None


def parse_json_column(text: str, key: str = "name") -> list:
    """Safely extracts a list of values from JSON string representation."""
    items = []
    try:
        data = ast.literal_eval(text)
        for item in data:
            if isinstance(item, dict) and key in item:
                items.append(item[key])
    except Exception:
        pass
    return items


def extract_top_cast(text: str, limit: int = 3) -> list:
    """Extracts top N cast members from cast JSON string."""
    cast = []
    try:
        data = ast.literal_eval(text)
        for idx, member in enumerate(data):
            if idx >= limit:
                break
            if isinstance(member, dict) and "name" in member:
                cast.append(member["name"])
    except Exception:
        pass
    return cast


def extract_director(text: str) -> list:
    """Extracts Director name from crew JSON string."""
    directors = []
    try:
        data = ast.literal_eval(text)
        for member in data:
            if isinstance(member, dict) and member.get("job") == "Director":
                directors.append(member["name"])
                break
    except Exception:
        pass
    return directors


def collapse_spaces(names_list: list) -> list:
    """Removes whitespace within entities (e.g., 'Sam Worthington' -> 'SamWorthington')."""
    return [name.replace(" ", "") for name in names_list]


def stem_text(text: str, stemmer: Optional[Any] = None) -> str:
    """Applies Porter Stemmer to tokenize and reduce words to root form if available."""
    if HAS_NLTK and stemmer:
        tokens = [stemmer.stem(word) for word in text.split()]
        return " ".join(tokens)
    return text


def build_recommender_pipeline(
    movies_path: str,
    credits_path: str,
    output_dir: str = ".",
    max_features: int = 5000,
    dtype: np.dtype = np.float32,
) -> None:
    """
    Executes full pipeline:
    1. Loads raw CSV datasets
    2. Merges datasets on title
    3. Cleans, parses, and extracts metadata (genres, keywords, cast, crew)
    4. Performs NLP stemming and feature vectorization
    5. Computes pairwise Cosine Similarity
    6. Saves artifacts: movie_dict.pkl and similarity.pkl
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading datasets from {movies_path} and {credits_path}...")
    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)

    print("[+] Merging datasets on 'title'...")
    df = movies_df.merge(credits_df, on="title")

    # Select relevant features
    df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    df.dropna(inplace=True)

    print("[+] Parsing metadata and entities...")
    df["genres"] = df["genres"].apply(parse_json_column)
    df["keywords"] = df["keywords"].apply(parse_json_column)
    df["cast"] = df["cast"].apply(lambda x: extract_top_cast(x, limit=3))
    df["crew"] = df["crew"].apply(extract_director)
    df["overview"] = df["overview"].apply(lambda x: x.split() if isinstance(x, str) else [])

    # Remove internal spaces to create unique tokens
    for col in ["genres", "keywords", "cast", "crew"]:
        df[col] = df[col].apply(collapse_spaces)

    # Combine all tags
    print("[+] Assembling comprehensive tags column...")
    df["tags"] = df["overview"] + df["genres"] + df["keywords"] + df["cast"] + df["crew"]

    final_df = df[["movie_id", "title", "tags"]].copy()
    final_df["tags"] = final_df["tags"].apply(lambda x: " ".join(x).lower())

    # NLP Stemming
    print("[+] Applying NLTK Porter Stemming...")
    stemmer = PorterStemmer() if HAS_NLTK else None
    final_df["tags"] = final_df["tags"].apply(lambda x: stem_text(x, stemmer))

    # Text Vectorization
    print(f"[+] Vectorizing tags using CountVectorizer (max_features={max_features})...")
    cv = CountVectorizer(max_features=max_features, stop_words="english")
    vectors = cv.fit_transform(final_df["tags"]).toarray()

    # Similarity Calculation
    print("[+] Calculating pairwise Cosine Similarity matrix...")
    similarity = cosine_similarity(vectors).astype(dtype)

    # Exporting
    movie_dict_path = out_dir / "movie_dict.pkl"
    similarity_path = out_dir / "similarity.pkl"

    print(f"[+] Saving movie metadata dictionary to {movie_dict_path}...")
    with open(movie_dict_path, "wb") as f:
        pickle.dump(final_df.to_dict(), f)

    print(f"[+] Saving similarity matrix to {similarity_path} (shape: {similarity.shape})...")
    with open(similarity_path, "wb") as f:
        pickle.dump(similarity, f)

    print("[✓] Pipeline execution finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Movie Recommender System")
    parser.add_argument("--movies", default="tmdb_5000_movies.csv", help="Path to tmdb_5000_movies.csv")
    parser.add_argument("--credits", default="tmdb_5000_credits.csv", help="Path to tmdb_5000_credits.csv")
    parser.add_argument("--output-dir", default=".", help="Directory to save output pickle files")
    parser.add_argument("--max-features", type=int, default=5000, help="Max vector features")
    args = parser.parse_args()

    # Note: If datasets are not present locally, user can download from Kaggle
    if Path(args.movies).exists() and Path(args.credits).exists():
        build_recommender_pipeline(
            movies_path=args.movies,
            credits_path=args.credits,
            output_dir=args.output_dir,
            max_features=args.max_features,
        )
    else:
        print("[!] Input CSV files not found. The pipeline script is ready to run whenever TMDB CSVs are supplied.")
        print("    Download dataset: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
