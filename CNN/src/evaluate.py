import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
from data import PLOT_DIR

# METRICS
def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    roc_auc = roc_auc_score(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

# PLOT HISTORY
def plot_history(history_head, history_fine, model_name):
    acc = history_head.history.get("accuracy", []) + history_fine.history.get("accuracy", [])
    val_acc = history_head.history.get("val_accuracy", []) + history_fine.history.get("val_accuracy", [])
    loss = history_head.history.get("loss", []) + history_fine.history.get("loss", [])
    val_loss = history_head.history.get("val_loss", []) + history_fine.history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    plt.plot(acc, label="train_accuracy")
    plt.plot(val_acc, label="val_accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_accuracy.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_loss.png"))
    plt.close()
