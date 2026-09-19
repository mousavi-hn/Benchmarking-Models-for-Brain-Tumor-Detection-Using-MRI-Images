"""Compute binary-classification metrics and predictions for hybrid models."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def calculate_metrics(y_true, y_prob, threshold=0.5):
    """Calculate binary classification performance metrics.
    
    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        threshold: Probability threshold used to obtain binary predictions.
    
    Returns:
        dict: Accuracy, precision, sensitivity/recall, specificity, F1, ROC-AUC,
            and confusion-matrix counts.
    """
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

def predict_on_sequence(model, sequence):
    """Run inference over an entire data sequence.
    
    Args:
        model: Trained Keras model.
        sequence: Iterable yielding image batches and labels.
    
    Returns:
        tuple: Ground-truth labels and predicted probabilities as NumPy arrays.
    """
    probs = []
    labels = []

    for batch_x, batch_y in sequence:
        batch_prob = model.predict(batch_x, verbose=0).ravel()
        probs.extend(batch_prob.tolist())
        labels.extend(batch_y.ravel().tolist())

    return np.asarray(labels, dtype=np.int32), np.asarray(probs, dtype=np.float32)