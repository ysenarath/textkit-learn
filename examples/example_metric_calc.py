from tklearn.metrics.base import Evaluator
from tklearn.metrics.classification import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    RocAuc,
)

if __name__ == "__main__":
    evaluator = Evaluator(
        metrics={
            "accuracy": Accuracy(),
            "precision": Precision(),
            "recall": Recall(),
            "auc": RocAuc(),
            "f1": F1Score(),
        }
    )
    evaluator.update_state(
        y_true=[0, 1, 0],
        y_pred=[0, 1, 1],
        sample_weight=[1, 1, 1],
        y_score=[0.1, 0.9, 0.8],
    )
    evaluator.update_state(
        y_true=[0, 1, 0],
        y_pred=[0, 1, 1],
        sample_weight=[1, 1, 1],
        y_score=[0.1, 0.9, 0.8],
    )
    results = evaluator.result(return_dict=True)
    print(results)
