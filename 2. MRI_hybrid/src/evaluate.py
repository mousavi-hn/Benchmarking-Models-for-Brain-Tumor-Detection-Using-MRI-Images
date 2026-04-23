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
import numpy as np
from data import PLOT_DIR

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
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# PLOTS
def plot_history(history_head, history_fine, model_tag):
    acc = history_head.history.get("accuracy", []) + history_fine.history.get("accuracy", [])
    val_acc = history_head.history.get("val_accuracy", []) + history_fine.history.get("val_accuracy", [])
    loss = history_head.history.get("loss", []) + history_fine.history.get("loss", [])
    val_loss = history_head.history.get("val_loss", []) + history_fine.history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    plt.plot(acc, label="train_accuracy")
    plt.plot(val_acc, label="val_accuracy")
    plt.title(f"{model_tag} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model_tag}_accuracy.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title(f"{model_tag} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model_tag}_loss.png"))
    plt.close()

# EVALUATION
def predict_on_sequence(model, sequence):
    probs = []
    labels = []

    for batch_x, batch_y in sequence:
        batch_prob = model.predict(batch_x, verbose=0).ravel()
        probs.extend(batch_prob.tolist())
        labels.extend(batch_y.ravel().tolist())

    return np.asarray(labels, dtype=np.int32), np.asarray(probs, dtype=np.float32)