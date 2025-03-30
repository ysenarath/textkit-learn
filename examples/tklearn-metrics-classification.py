from tklearn.metrics import MetricState
from tklearn.metrics.classification import (
    AUC,
    F1,
    Accuracy,
    Precision,
    Recall,
)

# Create a collection of metrics with names
metrics = MetricState({
    "accuracy": Accuracy(),
    "f1": F1(),
    "precision": Precision(),
    "recall": Recall(),
    "auc": AUC(),
})

# Example data
y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 1, 1, 0]
y_score = [0.1, 0.9, 0.2, 0.8, 0.51, 0.49]

# Update metrics with the data
metrics.update(y_true=y_true, y_pred=y_pred, y_score=y_score)

# Get the results
results = metrics.result()
print(results)
# {
#     "accuracy": 0.6666666666666666,
#     "f1": 0.6666666666666666,
#     "precision": 0.6666666666666666,
#     "recall": 0.6666666666666666,
#     "auc": 0.6666666666666667,
# }

# Define example data
y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 1, 1, 0]
y_score = [0.1, 0.9, 0.2, 0.8, 0.51, 0.49]

# Process in batches
for part in range(0, len(y_true), 2):
    batch_y_true = y_true[part : part + 2]
    batch_y_pred = y_pred[part : part + 2]
    batch_y_score = y_score[part : part + 2]

    # Update metrics with each batch
    metrics.update(
        y_true=batch_y_true,
        y_pred=batch_y_pred,
        y_score=batch_y_score,
    )

# Get final results after all batches
final_results = metrics.result()
print(final_results)
