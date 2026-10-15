"""Entrenamiento de la RNN.

Uso:  python -m src.train [--epochs 15] [--data-dir RUTA/aclImdb]
"""
import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from . import config
from .data import download_dataset, find_dataset_root, load_all, make_dataset
from .model import build_model
from .plots import plot_history


def main(epochs: int = config.EPOCHS, data_dir: str | None = None):
    tf.keras.utils.set_random_seed(config.SEED)
    for d in (config.MODELS_DIR, config.REPORTS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    root = find_dataset_root(Path(data_dir)) if data_dir else download_dataset()
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_all(root)
    print(f"Reseñas: train={len(x_tr)} val={len(x_va)} test={len(x_te)}")

    train_ds = make_dataset(x_tr, y_tr, training=True)
    val_ds = make_dataset(x_va, y_va, training=False)

    model = build_model(x_tr)
    model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=1, min_lr=1e-5, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    model.save(config.MODEL_PATH)   # EarlyStopping ya restauró los mejores pesos
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    config.HISTORY_PATH.write_text(json.dumps(hist, indent=2))
    plot_history(hist, config.FIGURES_DIR / "curvas_entrenamiento.png")
    print(f"Modelo guardado en {config.MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--data-dir", type=str, default=None, help="Carpeta aclImdb ya descargada")
    a = ap.parse_args()
    main(a.epochs, a.data_dir)
