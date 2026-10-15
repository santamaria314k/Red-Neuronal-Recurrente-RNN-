"""Evaluación sobre el conjunto de PRUEBA oficial (25.000 reseñas nunca vistas).

Uso:  python -m src.evaluate [--data-dir RUTA/aclImdb]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from tensorflow import keras

from . import config
from .data import download_dataset, find_dataset_root, load_all
from .plots import plot_confusion, plot_roc


def main(data_dir: str | None = None):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    root = find_dataset_root(Path(data_dir)) if data_dir else download_dataset()
    _, _, (x_te, y_te) = load_all(root)

    model = keras.models.load_model(config.MODEL_PATH)
    import tensorflow as tf
    probs = model.predict(tf.constant(x_te), batch_size=256, verbose=0).ravel()
    y_pred = (probs >= config.THRESHOLD).astype(int)

    auc = roc_auc_score(y_te, probs)
    report = classification_report(y_te, y_pred, target_names=config.CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    print(classification_report(y_te, y_pred, target_names=config.CLASS_NAMES, zero_division=0))
    metrics = {
        "test_accuracy": float(accuracy_score(y_te, y_pred)),
        "precision_positivo": float(precision_score(y_te, y_pred)),
        "recall_positivo": float(recall_score(y_te, y_pred)),
        "f1_positivo": float(f1_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "roc_auc": float(auc),
        "n_test": int(len(y_te)),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "classification_report": report,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    plot_confusion(y_te, y_pred, config.CLASS_NAMES, config.FIGURES_DIR / "matriz_confusion.png")
    plot_roc(y_te, probs, auc, config.FIGURES_DIR / "curva_roc.png")

    # 12 predicciones de ejemplo
    idx = np.random.default_rng(config.SEED).choice(len(x_te), 12, replace=False)
    rows = [{"texto": x_te[i][:110] + "...", "real": config.CLASS_NAMES[y_te[i]],
             "predicho": config.CLASS_NAMES[y_pred[i]],
             "prob_positivo": round(float(probs[i]), 4),
             "acierto": bool(y_pred[i] == y_te[i])} for i in idx]
    pd.DataFrame(rows).to_csv(config.PREDICTIONS_CSV, index=False)
    print(f"Accuracy test: {metrics['test_accuracy']:.4f} | F1: {metrics['f1_positivo']:.4f} | AUC: {auc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=None)
    main(ap.parse_args().data_dir)
