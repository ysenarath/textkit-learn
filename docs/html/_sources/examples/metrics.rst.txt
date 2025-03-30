# Metrics Library Documentation

=======================
Metrics Library
=======================

Overview
--------

This module provides a flexible and extensible framework for defining and managing metrics, particularly in a context-managed environment. It allows for creating custom metrics that can maintain state, be reset, updated, and return results.

Classes
-------

MetricBase
~~~~~~~~~~

An abstract base class that serves as the foundation for all metrics.

Methods
^^^^^^^

- ``reset()`` - Resets the metric state.
- ``update(**kwargs)`` - Updates the metric with new data.
- ``result()`` - An abstract method that must be implemented by subclasses to return the metric value.
- ``copy(deep=True)`` - Creates a copy of the metric.

  - ``deep`` - When True, performs a deep copy using cloudpickle.
- ``__call__(**kwargs)`` - Calls the metric with the provided arguments and returns the result.
- ``__repr__()`` - Returns a string representation of the metric.

MetricVariable
~~~~~~~~~~~~~

A descriptor class that allows for storing state variables within metrics.

Methods
^^^^^^^

- ``__set_name__(owner, name)`` - Sets the name of the variable when it's defined in a class.
- ``__get__(instance, owner)`` - Gets the value of the variable from the current metric context.
- ``__set__(instance, value)`` - Sets the value of the variable in the current metric context.

MetricState
~~~~~~~~~~~

A class that manages the state of multiple metrics, implementing both ``MetricBase`` and ``Mapping`` interfaces.

Initialization
^^^^^^^^^^^^^

.. code-block:: python

    def __init__(
        self,
        metrics=None,
        metric_names=None
    )

- ``metrics`` - Can be a single metric, a list of metrics, or a dictionary mapping names to metrics.
- ``metric_names`` - Optional list of names for metrics when provided as a list.

Methods
^^^^^^^

- ``add_metric(metric)`` - Adds a new metric to be tracked.
- ``reset()`` - Resets all tracked metrics.
- ``update(**kwargs)`` - Updates all tracked metrics with new data.
- ``result()`` - Returns a tuple of results or dictionary mapping names to results.
- Standard mapping methods (``__getitem__``, ``__setitem__``, ``__delitem__``, ``__iter__``, ``__len__``).

Functions
---------

with_metric_context
~~~~~~~~~~~~~~~~~~

A decorator that ensures a method runs within the appropriate metric context.

Usage Example
------------

.. code-block:: python

    # Define a custom metric by inheriting from MetricBase
    class AccuracyMetric(MetricBase):
        correct = MetricVariable[int]()
        total = MetricVariable[int]()

        def reset(self):
            self.correct = 0
            self.total = 0

        def update(self, y_true, y_pred):
            self.correct += sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
            self.total += len(y_true)

        def result(self):
            return self.correct / self.total if self.total > 0 else 0.0

    # Use the metric
    accuracy = AccuracyMetric()
    accuracy.update(y_true=[1, 0, 1], y_pred=[1, 1, 1])
    print(accuracy.result())  # 0.6666...

    # Use multiple metrics
    metrics = MetricState({
        'accuracy': AccuracyMetric(),
        'precision': PrecisionMetric()  # Another custom metric
    })
    metrics.update(y_true=[1, 0, 1], y_pred=[1, 1, 1])
    results = metrics.result()
    print(results['accuracy'])  # 0.6666...

Notes
-----

- The module uses context variables to manage metric state, which allows for thread-safe operation.
- Metrics can be composed by having metrics reference other metrics.
- State management is handled automatically through the ``MetricVariable`` descriptor and ``with_metric_context`` decorator.
