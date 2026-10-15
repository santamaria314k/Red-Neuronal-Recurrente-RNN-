"""Limpieza de texto y utilidades compartidas por entrenamiento, evaluación y app."""
import re
import string

_TAGS = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")
_PUNCT = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Quita etiquetas HTML (p. ej. <br />) y normaliza espacios.
    El paso a minúsculas y la eliminación de signos de puntuación los hace la capa
    TextVectorization dentro del modelo."""
    text = _TAGS.sub(" ", str(text))
    return _SPACES.sub(" ", text).strip()


def tokens_like_layer(text: str):
    """Tokeniza igual que TextVectorization (minúsculas, sin puntuación, split por espacios)."""
    return text.lower().translate(_PUNCT).split()


def oov_ratio(text: str, vocab: set) -> float:
    """Proporción de palabras del texto que no están en el vocabulario del modelo."""
    tokens = tokens_like_layer(text)
    if not tokens:
        return 1.0
    return sum(t not in vocab for t in tokens) / len(tokens)
