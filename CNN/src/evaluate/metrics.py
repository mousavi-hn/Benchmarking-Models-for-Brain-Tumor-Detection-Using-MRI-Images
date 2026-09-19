"""Compute binary-classification metrics for classical cancer-detection models."""
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