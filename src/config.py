"""Configuración central del proyecto (una sola fuente de verdad)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "sentiment_rnn.keras"
HISTORY_PATH = REPORTS_DIR / "history.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PREDICTIONS_CSV = REPORTS_DIR / "predicciones_12.csv"
RESULTS_MD = REPORTS_DIR / "RESULTADOS.md"

# IMDB Large Movie Review Dataset (Maas et al., 2011)
DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# Hiperparámetros
MAX_TOKENS = 20000      # tamaño del vocabulario
SEQ_LEN = 250           # longitud fija de cada secuencia (en tokens)
EMBED_DIM = 64
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 42
VAL_FRAC = 0.20         # fracción del train oficial que se usa como validación

CLASS_NAMES = ["negativo", "positivo"]   # etiqueta 0 / 1
THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.65   # por debajo: la app avisa que la predicción es incierta
OOV_WARN = 0.35               # por encima: demasiadas palabras fuera del vocabulario
