:github_url: https://github.com/sandylaker/saliency-metrics

.. _metrics:

Metrics
=======

.. currentmodule:: saliency_metrics.metrics


.. autosummary::
    :nosignatures:

    build_perturbation
    build_metric
    Perturbation
    SerializableResult
    ReInferenceMetric
    ReTrainingMetric
    AttributionMethod

Building Functions
------------------

build_perturbation
~~~~~~~~~~~~~~~~~~
.. autofunction:: build_perturbation

build_metric
~~~~~~~~~~~~
.. autofunction:: build_metric

General Protocols and Base Classes
----------------------------------

Perturbation
~~~~~~~~~~~~
.. autoprotocol:: Perturbation

Serializable Result
~~~~~~~~~~~~~~~~~~~
.. autoprotocol:: SerializableResult

Re-inference Based Metric
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoprotocol:: ReInferenceMetric

Re-training Based Metric
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoprotocol:: ReTrainingMetric

Attribution Method
~~~~~~~~~~~~~~~~~~
.. autoprotocol:: AttributionMethod
