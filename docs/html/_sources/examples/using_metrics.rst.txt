=================
Using Metrics
=================

Overview
========

This guide demonstrates how to use the tklearn metrics module for evaluating machine learning models. The framework provides a flexible way to compute multiple metrics simultaneously, accumulate results across batches, and generate performance visualizations.

Basic Usage
==========

Setting Up Metrics
-----------------

First, import the necessary metrics and create a ``MetricState`` to manage them:

.. code-block:: python

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

Computing Metrics
----------------

You can update metrics with your prediction data:

.. code-block:: python

   # Example data
   y_true = [0, 1, 1, 0, 1, 0]
   y_pred = [0, 1, 0, 1, 1, 0]
   y_score = [0.1, 0.9, 0.2, 0.8, 0.51, 0.49]

   # Update metrics with the data
   metrics.update(
       y_true=y_true,
       y_pred=y_pred,
       y_score=y_score
   )

   # Get the results
   results = metrics.result()
   print(results)

The results will be returned as a dictionary with the metric names as keys:

.. code-block:: python

   {
      "accuracy": 0.6666666666666666,
      "f1": 0.6666666666666666,
      "precision": 0.6666666666666666,
      "recall": 0.6666666666666666,
      "auc": 0.6666666666666667,
   }

Batch Processing
===============

One key advantage of the metrics framework is the ability to accumulate results across batches:

.. code-block:: python

   # Define example data
   y_true = [0, 1, 1, 0, 1, 0]
   y_pred = [0, 1, 0, 1, 1, 0]
   y_score = [0.1, 0.9, 0.2, 0.8, 0.51, 0.49]
   sample_weight = [1, 1, 1, 1, 1, 1]

   # Process in batches
   for part in range(0, len(y_true), 2):
       batch_y_true = y_true[part : part + 2]
       batch_y_pred = y_pred[part : part + 2]
       batch_y_score = y_score[part : part + 2]
       batch_sample_weight = sample_weight[part : part + 2]

       # Update metrics with each batch
       metrics.update(
           y_true=batch_y_true,
           y_pred=batch_y_pred,
           y_score=batch_y_score,
           sample_weight=batch_sample_weight,
       )

   # Get final results after all batches
   final_results = metrics.result()

This pattern is especially useful for evaluation during training or when working with large datasets that cannot fit into memory.

Advanced Metrics
===============

Optimal Thresholds
-----------------

The framework includes special metrics for finding optimal classification thresholds:

.. code-block:: python

   from tklearn.metrics.classification import OptimalPRThreshold, OptimalAUCThreshold

   # Find the optimal threshold based on Precision-Recall curve
   pr_curve = OptimalPRThreshold()
   pr_data = pr_curve(y_true=y_true, y_score=y_score)

   # The optimal threshold value
   optimal_threshold = pr_data["threshold"]
   print(f"Optimal PR threshold: {optimal_threshold}")

   # Similarly for ROC curve
   roc_curve = OptimalAUCThreshold()
   roc_data = roc_curve(y_true=y_true, y_score=y_score)
   optimal_roc_threshold = roc_data["threshold"]
   print(f"Optimal ROC threshold: {optimal_roc_threshold}")

Visualization
============

The metrics framework provides data for visualizing performance curves:

Precision-Recall Curve
---------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   from sklearn.metrics import PrecisionRecallDisplay

   # Get PR curve data
   pr_data = OptimalPRThreshold()(y_true=y_true, y_score=y_score)
   plot_data = pr_data["data"]

   # Extract precision and recall values
   precision = [item["precision"] for item in plot_data]
   recall = [item["recall"] for item in plot_data]

   # Create the visualization
   display = PrecisionRecallDisplay(precision=precision, recall=recall)
   display.plot()
   plt.title("Precision-Recall Curve")
   plt.show()

ROC Curve
--------

.. code-block:: python

   # Get ROC curve data
   roc_data = OptimalAUCThreshold()(y_true=y_true, y_score=y_score)
   plot_data = roc_data["data"]

   # Extract FPR and TPR values
   fpr = [item["fpr"] for item in plot_data]
   tpr = [item["tpr"] for item in plot_data]

   # Create the visualization
   plt.figure()
   plt.plot(fpr, tpr, lw=2, label="ROC curve")
   plt.plot([0, 1], [0, 1], 'k--', lw=2)
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic')
   plt.legend(loc="lower right")
   plt.show()

Sample Weights
=============

Many metrics support sample weighting to give different importance to different samples:

.. code-block:: python

   # Create sample weights (e.g., to emphasize certain samples)
   sample_weight = [2.0, 1.0, 0.5, 1.0, 1.0, 0.8]

   # Update metrics with weights
   metrics.update(
       y_true=y_true,
       y_pred=y_pred,
       y_score=y_score,
       sample_weight=sample_weight
   )

   # Results now reflect the weighted importance of samples
   weighted_results = metrics.result()

Complete Example
===============

Here's a complete example demonstrating metric computation and visualization:

.. code-block:: python

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
       # Create metrics collection
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
       sample_weight = [1, 1, 1, 1]

       # Process in batches
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

       # Print metric results
       print(metrics.result())

       # Calculate optimal PR threshold and visualize PR curve
       pr_curve = OptimalPRThreshold()
       pr_data = pr_curve(y_true=y_true, y_score=y_score)
       pprint(f"Optimal PR threshold: {pr_data['threshold']}")

       plot_data = pr_data["data"]
       precision = [plot_data_item["precision"] for plot_data_item in plot_data]
       recall = [plot_data_item["recall"] for plot_data_item in plot_data]

       plot = PrecisionRecallDisplay(precision=precision, recall=recall)
       plot.plot()
       plt.title("Precision-Recall Curve")
       plt.show()

       # Calculate optimal AUC threshold and visualize ROC curve
       roc_data = OptimalAUCThreshold()(y_true=y_true, y_score=y_score)
       fpr = [item["fpr"] for item in roc_data["data"]]
       tpr = [item["tpr"] for item in roc_data["data"]]

       plt.figure()
       plt.plot(fpr, tpr, color="b", label="ROC curve", lw=2, alpha=0.8)
       plt.plot([0, 1], [0, 1], 'k--', lw=2)
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title('Receiver Operating Characteristic')
       plt.legend(loc="lower right")
       pprint(f"Optimal AUC threshold: {roc_data['threshold']}")
       plt.show()

   if __name__ == "__main__":
       main()

See Also
========

- :doc:`creating_metrics`: Documentation for creating custom metrics
- ``sklearn.metrics``: Scikit-learn's metrics module, which provides the foundation for many tklearn metrics

Tips and Best Practices
======================

1. **Reset Metrics**: To reuse metrics for a new evaluation, call ``metrics.reset()``
2. **Sample Weighting**: Use sample weights to handle class imbalance or give importance to certain examples
3. **Batch Processing**: Process large datasets in batches to manage memory efficiently
4. **Visualization**: Use the data from optimal threshold metrics to create visualizations of model performance
5. **Custom Metrics**: Extend the framework with your own custom metrics by inheriting from ``MetricBase``
