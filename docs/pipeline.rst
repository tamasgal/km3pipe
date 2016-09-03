Pipeline Workflow
=================

KM3Pipe is a basic framework which tries to give you a lose structure and
workflow for data analysis. It has a simple, yet powerful module system
which allows you to organise and reuse code.

The main structure is a ``Pipeline`` which is meant to hold everything
together. The building blocks are simply called ``Modules``.

To setup a workflow, you first create a pipeline, attach the modules to it
and to fire up the analysis chain, you call ``.drain()`` on your pipeline
and let the flow go.


The following script shows the module system of km3pipe.
There is a ``Pump`` which is in this case a dummy data generator. The other
Modules do some modifications on the data and pass them through to the next
module in the pipeline.

.. literalinclude:: ../examples/module_workflow.py
   :language: python
      :linenos:

Modules
-------

A module is a configurable building block which can be attached to a pipeline.
It has a ``process()`` method, which is called every time with the current
data ("blob") in the pipeline cycle. This piece of data can be analysed,
manipulated and finally returned to allow the handover to the next module
in the pipeline system.

Instance variables can be initialised within the ``__init__()`` method.
User defined parameters are accessible via the ``get()`` method, which either
returns the actual value or ``None`` if not defined.
This allows an easy way to define default values as seen in the example below.

.. literalinclude:: ../examples/module_workflow.py
   :pyobject: Foo
   :emphasize-lines: 5-6
   :linenos:

To override the default parameters, the desired values can be set when
attaching the module to the pipeline. Always use the class itself, since
the ``attach()`` method of the pipeline will care about the initialisation::

    pipe.attach(Foo, 'foo_module', foo='dummyfoo', bar='dummybar')
