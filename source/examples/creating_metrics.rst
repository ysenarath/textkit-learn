===================
Creating New Metrics
===================

Overview
========

The tklearn metrics framework provides a standardized way to implement performance metrics for machine learning models. The framework handles the accumulation of data and calculation of metric values through a well-defined interface.

Base Classes
===========

The metrics framework consists of three main components:

- ``MetricBase``: The abstract base class for all metrics
- ``MetricState``: Manages the state of metrics during computation
- ``MetricVariable``: A descriptor class for storing metric state variables

Creating a Custom Metric
=======================

1. Inherit from MetricBase
-------------------------

Your custom metric should inherit from the ``MetricBase`` abstract class:

.. code-block:: python

   from tklearn.metrics.base import MetricBase
   from tklearn.metrics.helpers import ArrayAccum

   class MyMetric(MetricBase):
       def __init__(self, param1=None, param2=False):
           super().__init__()
           self.param1 = param1
           self.param2 = param2


2. Define State Variables
------------------------

State variables are used to accumulate data for the metric calculation. Use ``ArrayAccum`` or create custom accumulators:

.. code-block:: python

   class MyMetric(MetricBase):
       # Array accumulator for storing true values
       y_true = ArrayAccum("y_true")

       # Array accumulator for storing predicted values
       y_pred = ArrayAccum("y_pred")

       # Optional weight accumulator
       sample_weight = ArrayAccum("sample_weight")

.. note::
   The ``ArrayAccum`` helper class is used to accumulate data in a memory-efficient way. It automatically handles the conversion of data to NumPy arrays.


3. Implement Required Methods
----------------------------

All metrics must implement the ``result()`` method, which calculates and returns the final metric value:

.. code-block:: python

   def result(self) -> torch.Tensor:
       y_true = self.y_true.result()
       y_pred = self.y_pred.result()

       # Compute the metric
       result_value = compute_my_metric(y_true, y_pred, self.param1, self.param2)

       return result_value

The ``reset()`` and ``update()`` methods are inherited from ``MetricBase`` and generally don't need to be overridden.

Example Implementation
=====================

Here's a complete implementation of a custom accuracy metric:

.. code-block:: python

   from tklearn.metrics.base import MetricBase
   from tklearn.metrics.helpers import ArrayAccum
   import torch
   import numpy as np

   class BinaryAccuracy(MetricBase):
       y_true = ArrayAccum("y_true")
       y_pred = ArrayAccum("y_pred")
       sample_weight = ArrayAccum("sample_weight")

       def __init__(self, threshold=0.5):
           super().__init__()
           self.threshold = threshold

       def result(self) -> torch.Tensor:
           y_true = self.y_true.result()
           y_pred = self.y_pred.result()
           sample_weight = self.sample_weight.result()

           # Apply threshold to get binary predictions
           y_pred_binary = (y_pred >= self.threshold).astype(np.int64)

           # Calculate accuracy
           correct = (y_true == y_pred_binary)

           if sample_weight is not None:
               return np.sum(correct * sample_weight) / np.sum(sample_weight)

           return np.mean(correct)

Using a Custom Metric
====================

Once implemented, your custom metric can be used as follows:

.. code-block:: python

   # Create an instance of your metric
   my_metric = MyCustomMetric(param1=10, param2=True)

   # Update the metric with data
   my_metric.update(y_true=true_values, y_pred=predicted_values)

   # Calculate the metric value
   result = my_metric.result()

You can also use metrics in a batch processing context:

.. code-block:: python

   metric = MyCustomMetric()
   for batch in data_loader:
       # Get batch data
       x, y_true = batch

       # Get predictions
       y_pred = model(x)

       # Update the metric
       metric.update(y_true=y_true, y_pred=y_pred)

   # Calculate final result
   final_result = metric.result()

Combining Multiple Metrics
=========================

Use ``MetricState`` to manage multiple metrics together:

.. code-block:: python

   from tklearn.metrics.base import MetricState

   # Create metrics
   accuracy = Accuracy()
   precision = Precision()
   recall = Recall()

   # Combine metrics with names
   metrics = MetricState({
       'accuracy': accuracy,
       'precision': precision,
       'recall': recall
   })

   # Update all metrics at once
   metrics.update(y_true=y_true, y_pred=y_pred)

   # Get results
   results = metrics.result()  # Returns a dictionary with metric names as keys

Advanced Usage
=============

Copying Metrics
--------------

You can create a copy of a metric instance:

.. code-block:: python

   new_metric = my_metric.copy(deep=True)  # Create a deep copy

Direct Evaluation
---------------

Call the metric instance directly for one-shot evaluation:

.. code-block:: python

   result = my_metric(y_true=true_values, y_pred=predicted_values)

Best Practices
=============

1. Always call the parent class constructor with ``super().__init__()``
2. Document the parameters in the constructor and the return value of ``result()``
3. Verify that the input data is in the expected format
4. Handle edge cases (e.g., empty arrays, division by zero)
5. Use type hints to improve code readability and IDE integration
