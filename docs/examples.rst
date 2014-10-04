.. _examples:

Examples
========

KM3Pipe Workflow
----------------

KM3Pipe is a basic framework which tries to give you a lose structure and
workflow of data analysis. It has a simple, yet powerful module system
which allows you to organise and reuse your code.

The main structure is a ``Pipeline`` which is meant to hold everything
together. The building blocks are simply called ``Modules``.

To setup a workflow, you first create a pipeline, attach the modules to it
and to start, you call ``.drain()`` on your pipeline and let the flow go.


The following script shows the module system of km3pipe.
There is a ``Pump`` which is in this case a dummy data generator. The other
Modules do some modifications on the data and pass them through to the next
module in the pipeline.

.. literalinclude:: ../examples/module_workflow.py
   :language: python
   :linenos: