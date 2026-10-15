"""Predicción sobre texto nuevo (CLI y reutilizado por Streamlit).

Uso:  python -m src.predict "This movie was great"
"""
import sys

import tensorflow as tf
from tensorflow import keras

from . import config
from .text_utils import clean_text, oov_ratio, tokens_like_layer


def load_artifacts(model_path=config.MODEL_PATH):
    model = keras.models.load_model(model_path)
    vocab = set(model.get_layer("vectorizacion").get_vocabulary())
    return model, vocab


def predict_text(model, vocab, text: str) -> dict:
    cleaned = clean_text(text)
    p_pos = float(model(tf.constant([cleaned]), training=False).numpy()[0][0])
    positive = p_pos >= config.THRESHOLD
    return {
        "label": config.CLASS_NAMES[int(positive)],
        "confidence": p_pos if positive else 1.0 - p_pos,
        "prob_positive": p_pos,
        "prob_negative": 1.0 - p_pos,
        "n_tokens": len(tokens_like_layer(cleaned)),
        "oov_ratio": oov_ratio(cleaned, vocab),
        "cleaned": cleaned,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Uso: python -m src.predict "texto de la reseña"')
    m, v = load_artifacts()
    r = predict_text(m, v, sys.argv[1])
    print(f"Sentimiento: {r['label']}  |  Confianza: {r['confidence']*100:.2f}%")
    print(f"P(positivo) = {r['prob_positive']:.4f}  |  Palabras desconocidas: {r['oov_ratio']*100:.0f}%")
