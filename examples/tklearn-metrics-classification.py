from pprint import pprint

import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

from tklearn.metrics import MetricState
from tklearn.metrics.classification import (
    AUC,
    F1,
    Accuracy,
    OptimalAUCThreshold,
    OptimalPRThreshold,
    Precision,
    Recall,
)


def main():
    metrics = MetricState({
        "accuracy": Accuracy(),
        "f1": F1(),
        "precision": Precision(),
        "recall": Recall(),
        "auc": AUC(),
    })
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.51, 0.49]
    sample_weight = [1, 1, 1, 1]
    for part in range(0, 4, 2):
        batch_y_true = y_true[part : part + 2]
        batch_y_pred = y_pred[part : part + 2]
        batch_y_score = y_score[part : part + 2]
        batch_sample_weight = sample_weight[part : part + 2]
        metrics.update(
            y_true=batch_y_true,
            y_pred=batch_y_pred,
            y_score=batch_y_score,
            sample_weight=batch_sample_weight,
        )
    print(metrics.result())
    pr_curve = OptimalPRThreshold()
    pr_data = pr_curve(y_true=y_true, y_score=y_score)
    pprint(pr_data["threshold"])
    plot_data = pr_data["data"]
    precision = [plot_data_item["precision"] for plot_data_item in plot_data]
    recall = [plot_data_item["recall"] for plot_data_item in plot_data]
    #
    plot = PrecisionRecallDisplay(precision=precision, recall=recall)
    plot.plot()
    plt.show()
    #
    pr_data = OptimalAUCThreshold()(y_true=y_true, y_score=y_score)
    fpr = [item["fpr"] for item in pr_data["data"]]
    tpr = [item["tpr"] for item in pr_data["data"]]
    #
    ax = plt.gca()
    ax.plot(
        fpr,
        tpr,
        color="b",
        # label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        label="Mean ROC",
        lw=2,
        alpha=0.8,
    )
    pprint(pr_data["threshold"])


if __name__ == "__main__":
    main()
