"""TF-IDF document recommender (Path A).

Loads a corpus (corpus.csv with columns id, title, text, category),
cleans the text, vectorizes it with TF-IDF and recommends the Top-K most
similar documents to a seed document ID or to a free-text query using
cosine similarity.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus.csv"
REQUIRED_COLUMNS = ("id", "title", "text")
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class Recommendation:
    """One recommended document with its similarity score."""

    rank: int
    doc_id: str
    title: str
    category: str
    score: float


def clean_text(text: str) -> str:
    """Lowercase, strip accents and keep only letters, digits and spaces."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> pd.DataFrame:
    """Read the corpus CSV and validate its structure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    corpus = pd.read_csv(path, dtype=str).fillna("")
    missing = [col for col in REQUIRED_COLUMNS if col not in corpus.columns]
    if missing:
        raise ValueError(f"Corpus is missing required columns: {missing}")
    if "category" not in corpus.columns:
        corpus["category"] = ""
    if corpus["id"].duplicated().any():
        raise ValueError("Corpus contains duplicated document IDs.")
    corpus["id"] = corpus["id"].str.strip().str.upper()
    return corpus.reset_index(drop=True)


class TfidfRecommender:
    """Content-based recommender built on TF-IDF and cosine similarity."""

    def __init__(self, corpus: pd.DataFrame, use_stopwords: bool = True) -> None:
        self.corpus = corpus
        # Title is included so short documents still carry their main topic.
        documents = (corpus["title"] + " " + corpus["text"]).map(clean_text)
        self.vectorizer = TfidfVectorizer(
            stop_words=list(ENGLISH_STOP_WORDS) if use_stopwords else None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(documents)
        self._index = {doc_id: i for i, doc_id in enumerate(corpus["id"])}

    def has_id(self, doc_id: str) -> bool:
        return doc_id.strip().upper() in self._index

    def get_document(self, doc_id: str) -> pd.Series:
        doc_id = doc_id.strip().upper()
        if doc_id not in self._index:
            raise KeyError(f"Document ID not found: {doc_id}")
        return self.corpus.iloc[self._index[doc_id]]

    def recommend_by_id(self, doc_id: str, k: int = DEFAULT_TOP_K) -> list[Recommendation]:
        """Top-K documents most similar to a seed document (seed excluded)."""
        doc_id = doc_id.strip().upper()
        if doc_id not in self._index:
            raise KeyError(f"Document ID not found: {doc_id}")
        seed = self._index[doc_id]
        scores = cosine_similarity(self.matrix[seed], self.matrix).ravel()
        scores[seed] = -1.0  # exclude the seed document itself
        return self._top_k(scores, k)

    def recommend_by_text(self, query: str, k: int = DEFAULT_TOP_K) -> list[Recommendation]:
        """Top-K documents most similar to a free-text query."""
        cleaned = clean_text(query)
        if not cleaned:
            raise ValueError("The query text is empty after cleaning.")
        query_vector = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        return self._top_k(scores, k)

    def _top_k(self, scores, k: int) -> list[Recommendation]:
        order = scores.argsort()[::-1][:k]
        results = []
        for rank, i in enumerate(order, start=1):
            row = self.corpus.iloc[i]
            results.append(
                Recommendation(rank, row["id"], row["title"], row["category"], float(scores[i]))
            )
        return results


def format_results(results: list[Recommendation]) -> str:
    """Render recommendations as a console table."""
    lines = [f"{'#':<3}{'ID':<8}{'Score':<8}{'Category':<19}Title", "-" * 78]
    for r in results:
        lines.append(f"{r.rank:<3}{r.doc_id:<8}{r.score:<8.4f}{r.category:<19}{r.title}")
    return "\n".join(lines)
