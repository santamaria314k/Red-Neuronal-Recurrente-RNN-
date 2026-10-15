"""App Streamlit: escribe una reseña y obtén el sentimiento predicho + confianza.

Ejecutar desde la raíz del repo:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config
from src.predict import load_artifacts, predict_text

st.set_page_config(page_title="Sentimiento de reseñas (RNN)", page_icon="🎬")

EJEMPLOS = {
    "Ejemplo positivo": ("This movie was absolutely wonderful. The acting was brilliant, the story "
                         "kept me engaged from start to finish, and the ending was truly moving."),
    "Ejemplo negativo": ("A complete waste of time. The plot made no sense, the dialogue was painful "
                         "to listen to, and the acting was so bad that I could not wait for it to end."),
    "Ejemplo mixto": ("The visuals were stunning and the soundtrack was great, but the story was "
                      "slow and the characters were forgettable."),
}


@st.cache_resource(show_spinner="Cargando modelo...")
def get_model():
    return load_artifacts()


def set_text(value: str):
    st.session_state["texto"] = value


st.title("🎬 Análisis de sentimiento con una RNN (LSTM)")
st.write("Escribe una **reseña de película en inglés** y el modelo indicará si es "
         "**positiva** o **negativa**, con su nivel de confianza.")

if not config.MODEL_PATH.exists():
    st.error("No se encontró el modelo entrenado en `models/`. "
             "Ejecuta primero el notebook o `python -m src.train`.")
    st.stop()

model, vocab = get_model()

with st.sidebar:
    st.header("Información del modelo")
    st.write("LSTM bidireccional entrenada con reseñas de IMDB (inglés).")
    if config.METRICS_PATH.exists():
        m = json.loads(config.METRICS_PATH.read_text())
        st.metric("Accuracy en test", f"{m['test_accuracy']*100:.1f}%")
        st.metric("F1 (clase positiva)", f"{m['f1_positivo']*100:.1f}%")
        st.metric("AUC", f"{m['roc_auc']:.3f}")

st.write("**Ejemplos rápidos:**")
cols = st.columns(len(EJEMPLOS))
for col, (name, txt) in zip(cols, EJEMPLOS.items()):
    col.button(name, on_click=set_text, args=(txt,), use_container_width=True)

texto = st.text_area("Reseña", key="texto", height=180,
                     placeholder="Escribe aquí una reseña en inglés...")

if st.button("Analizar sentimiento", type="primary"):
    if not texto.strip():
        st.warning("Escribe una reseña primero.")
        st.stop()
    res = predict_text(model, vocab, texto)
    icon = "😀" if res["label"] == "positivo" else "😞"
    st.subheader(f"{icon} Sentimiento: {res['label'].capitalize()}")
    c1, c2 = st.columns(2)
    c1.metric("Confianza", f"{res['confidence']*100:.2f}%")
    c2.metric("P(positivo)", f"{res['prob_positive']*100:.2f}%")
    st.progress(min(max(res["prob_positive"], 0.0), 1.0), text="0 % = negativo  ·  100 % = positivo")

    if res["n_tokens"] < 5:
        st.warning("El texto es muy corto; la predicción puede no ser fiable.")
    if res["oov_ratio"] > config.OOV_WARN:
        st.warning(f"El {res['oov_ratio']*100:.0f}% de las palabras no está en el vocabulario del modelo. "
                   "Fue entrenado con reseñas en inglés; en otro idioma la predicción no es fiable.")
    if res["confidence"] < config.CONFIDENCE_THRESHOLD:
        st.info("Confianza baja: la reseña puede ser ambigua o mezclar opiniones.")

    st.subheader("Probabilidad por clase")
    df = pd.DataFrame({"Clase": ["Negativo", "Positivo"],
                       "Probabilidad (%)": [res["prob_negative"] * 100, res["prob_positive"] * 100]}
                      ).set_index("Clase")
    st.bar_chart(df)
    with st.expander("Texto después de la limpieza"):
        st.write(res["cleaned"])
