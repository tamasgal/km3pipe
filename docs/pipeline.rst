Pipeline Workflow
=================

.. contents:: :local:

KM3Pipe is a lightweight framework which tries to give you a lose structure and
workflow for data analysis. It has a simple, yet powerful module system
which allows you to organise and reuse code.

The main structure is a ``Pipeline`` which is meant to hold everything
together. The building blocks are simply called Modules and are either
basic Python functions or instances of the class ``Module``.

To setup a workflow, you first create a pipeline, attach the modules to it
and to fire up the analysis chain, you call ``.drain()`` on your pipeline
and let the flow go.


The following script shows the module system of KM3Pipe.
There is a ``Pump`` which is in this case a dummy data generator. The other
Modules do some modifications on the data and pass them through to the next
module in the pipeline.

.. literalinclude:: ../examples/nogallery/module_workflow.py
   :language: python
      :linenos:

Modules
-------

A module is a configurable building block which can be attached to a pipeline.
It has a ``process()`` method, which is called every time with the current
data ("blob") in the pipeline cycle. This piece of data can be analysed,
manipulated and finally returned to allow the handover to the next module
in the pipeline system.

Instance variables can be initialised within the ``configure()`` method.
User defined parameters are accessible via the ``get()`` or ``required()``
method. Both of them return the passed value or ``None`` if not defined.
This allows an easy way to define default values as seen in the example below.

.. literalinclude:: ../examples/nogallery/module_workflow.py
   :pyobject: Foo
   :emphasize-lines: 4-6
   :linenos:

To override the default parameters, the desired values can be set when
attaching the module to the pipeline. Always use the class itself, since
the ``attach()`` method of the pipeline will care about the initialisation::

    pipe.attach(Foo, bar='dummybar', baz=69)


Pumps / Sinks
-------------

The pump and sink are special types of ``Module`` and are usually the
first and last ones to be attached to a pipeline. They are responsible
for reading and writing data to/from files, or streams from socket
connections.

``Pump`` and ``Sink`` inherits from the ``Module`` class. The
``__init__()`` method should be used to set up the file or socket
handler and the ``finish()`` has to close them. The actual data is
passed via the ``process()`` method. A data chunk is internally called
``Blob`` and usually represents an event.

To end the data pumping, the pump has to raise a ``StopIteration``
exception. One elegant way to implement this in Python is using a
generator.

The following example shows a very basic pump, which simply initialises
a list of dictionaries and "io" one blob after another on each
``process()`` call to the next module in the pipeline.


.. literalinclude:: ../examples/nogallery/module_workflow.py
   :pyobject: DummyPump
   :linenos:
