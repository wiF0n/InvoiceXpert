"""
This file contains the metrics used for evaluation.
"""

import evaluate
import numpy as np

# Define metrics
clf_metrics_list = ["accuracy", "f1", "precision", "recall"]
cfm_metric_name = "confusion_matrix"

clf_metrics = evaluate.combine(clf_metrics_list)
cfm_metric = evaluate.load(cfm_metric_name)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)
