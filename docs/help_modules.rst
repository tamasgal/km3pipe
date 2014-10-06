.. _help_modules:

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
This allows an easy way to define default values as seen in the example below.

.. literalinclude:: ../examples/module_workflow.py
   :pyobject: Foo
   :emphasize-lines: 5-6
   :linenos:

To override the default parameters, the desired values can be set when
attaching the module to the pipeline. Always use the class itself, since
the ``attach()`` method of the pipeline will care about the initialisation::

    pipe.attach(Foo, 'foo_module', foo='dummyfoo', bar='dummybar')
