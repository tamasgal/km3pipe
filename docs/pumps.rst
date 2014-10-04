.. _pumps:

Pumps
=====

The pump is usually the first module to be attached to a pipeline. It is
responsible for data generation by reading data files or streams from socket
connections.

``Pump`` inherits from the ``Module`` class. The ``__init__()`` method should
 be used to set up the file or socket handler and the ``finish()`` has to
 close them. The actual data is passed via the ``process()``method.

The following example shows a very basic pump, which simply initialises a
dictionary and "pumps" one entry after another on each ``process()``call.



.. literalinclude:: ../examples/module_workflow.py
   :pyobject: Pump
   :linenos: