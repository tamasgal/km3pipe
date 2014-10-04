.. _modules:

Modules
=======

A module is a configurable building block which can be attached to a pipeline.
It has a ``process()`` method, which is called every time with the current
data ("blob") in the pipeline cycle. This piece of data can be analysed,
manipulated and finally returned to allow the handover to the next module
in the pipeline system.

Instance variables can be initialised within the ``__init__()`` method.
User defined parameters are accessible via the ``get()`` method, which either
returns the actual value or ``None`` if not defined.
This allows an easy way to define default values

.. literalinclude:: ../examples/module_workflow.py
   :pyobject: Foo
   :emphasize-lines: 8-10
   :linenos:
