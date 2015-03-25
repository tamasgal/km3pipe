.. _help_pumps:

Pumps
=====

The pump is a special type of ``Module`` and is usually the first one to be
attached to a pipeline. It is responsible for data generation by reading data
files or streams from socket connections.

``Pump`` inherits from the ``Module`` class. The ``__init__()`` method should
be used to set up the file or socket handler and the ``finish()`` has to
close them. The actual data is passed via the ``process()`` method. A
data chunk is internally called ``Blob`` and usually represents an event.

To end the data pumping, the pump has to raise a ``StopIteration`` exception.
One elegant way to implement this in Python is using a generator.

The following example shows a very basic pump, which simply initialises a
list of dictionaries and "pumps" one blob after another on
each ``process()`` call to the next module in the pipeline.


.. literalinclude:: ../examples/module_workflow.py
   :pyobject: DummyPump
   :linenos:
