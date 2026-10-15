"""Descarga, lectura y partición del dataset IMDB (aclImdb)."""
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from . import config
from .text_utils import clean_text


def _is_root(d: Path) -> bool:
    return all((d / s / c).is_dir() for s in ("train", "test") for c in ("pos", "neg"))


def find_dataset_root(base: Path) -> Path:
    """Busca la carpeta aclImdb (con train/test y pos/neg); tolera extracciones anidadas."""
    base = Path(base)
    level = [base]
    for _ in range(4):                      # busca hasta 4 niveles de profundidad
        for cand in level:
            if _is_root(cand):
                return cand
        nxt = []
        for cand in level:
            try:
                nxt += [Path(e.path) for e in os.scandir(cand) if e.is_dir()]
            except OSError:
                pass
        level = nxt
    raise RuntimeError(f"No se encontró la carpeta aclImdb dentro de {base}. "
                       "Borra la carpeta 'data' y vuelve a ejecutar.")


def download_dataset(data_dir: Path = config.DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = tf.keras.utils.get_file(
        fname="aclImdb_v1", origin=config.DATASET_URL, untar=True,
        cache_dir=str(data_dir), cache_subdir="")
    extracted = Path(extracted)
    return find_dataset_root(extracted if extracted.is_dir() else data_dir)


def read_split(root: Path, split: str):
    """Lee train o test. Etiqueta 0 = neg (negativo), 1 = pos (positivo). Ignora 'unsup'."""
    texts, labels = [], []
    for label, name in enumerate(["neg", "pos"]):
        for p in sorted((Path(root) / split / name).glob("*.txt")):
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            labels.append(label)
    return texts, np.array(labels, dtype="int32")


def load_all(root: Path):
    """Devuelve (train, val, test) ya limpios. test = conjunto de prueba OFICIAL;
    val sale de una partición estratificada del train oficial."""
    tr_x, tr_y = read_split(root, "train")
    te_x, te_y = read_split(root, "test")
    tr_x = [clean_text(t) for t in tr_x]
    te_x = [clean_text(t) for t in te_x]
    x_tr, x_va, y_tr, y_va = train_test_split(
        tr_x, tr_y, test_size=config.VAL_FRAC, stratify=tr_y, random_state=config.SEED)
    return (x_tr, y_tr), (x_va, y_va), (te_x, te_y)


def make_dataset(texts, labels, training: bool, batch_size=config.BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((tf.constant(texts), labels.astype("float32")))
    if training:
        ds = ds.shuffle(min(len(texts), 20000), seed=config.SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
