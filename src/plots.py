"""Gráficas reutilizadas por train/evaluate."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve


def plot_history(history: dict, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Pérdida (loss)")):
        ax.plot(history[key], label="entrenamiento", marker="o")
        ax.plot(history["val_" + key], label="validación", marker="o")
        ax.set_title(title); ax.set_xlabel("Época"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_confusion(y_true, y_pred, class_names, out_path=None):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names,
                                            cmap="Blues", ax=ax, values_format="d")
    ax.set_title("Matriz de confusión (conjunto de prueba)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_roc(y_true, probs, auc, out_path=None):
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"RNN (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC (conjunto de prueba)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig
