Pipeline Workflow
=================

.. contents:: :local:

km3pipe is a lightweight framework which tries to give you a lose structure and
workflow for data analysis. It uses the simple, yet powerful module system
of the `thepipe <https://github.com/tamasgal/thepipe>`_ Python module,
which allows you to organise and reuse code.

In this section, a few examples are shown how to work the with
pipeline, but it is recommended to check the `documentation of thepipe
<https://thepipe.readthedocs.io/en/latest/?badge=latest>`_ since the features
will change in future.

The main structure is a ``Pipeline`` which is meant to hold everything
together. The building blocks are simply called Modules and are either
basic Python functions or instances of the class ``Module``. These classes
are simply imported from ``thepipe`` package and can also be used directly
from that.

To setup a workflow, you first create a pipeline, attach the modules to it
and to fire up the analysis chain, you call ``.drain()`` on your pipeline
and let the flow go.

Draining Procedure
------------------

Everything is standing still up until the ``pipe.drain()`` method is called.
This method can be called without arguments, which will cycle through the iterations
until the attached pump stops it. One can however also pass a number, which stands
for the maximum number of iterations, like ``pipe.drain(23)``.
Another possibility to stop the pipeline before the pump finishes is pressing ``CTRL+C``,
which will execute the current cycle, gracefully finish up every module and close
all the file handlers properly.

The stages a pipeline launches when ``pipe.drain()`` is executed are the following:

- Call ``.configure()`` on every attached module one-by-one
- Call ``.prepare()`` on every attached module one-by-one
- Start with the actual iteration and call ``.process(blob)`` on the first module
- Take it's output (a ``Blob()``) and call the next attached module with that Blob instance
- When the cycle reached the last module, start from the beginning
- Stop whenever (one of) the pump raises a ``StopIteration``
- Call ``.finish()`` on every module one-by-one


Pipeline Example
----------------

The following script shows the module system of KM3Pipe. There is a
``DummyPump`` which is in this case a dummy data generator. The other Modules
do some modifications on the data and pass them through to the next module in
the pipeline.

.. literalinclude:: ../examples/nogallery/module_workflow.py
   :language: python
   :linenos:



Which will print the following:::

    Pipeline and module initialisation took 0.000s (CPU 0.000s).
    This is the current blob: {'nr': 1}
    {'nr': 1, 'foo_entry': 'default_foo', 'moo_entry': 42}
    This is the current blob: {'nr': 2}
    {'nr': 2, 'foo_entry': 'default_foo', 'moo_entry': 42}
    My process() method was called 2 times.
    ============================================================
    2 cycles drained in 0.000553s (CPU 0.000525s). Memory peak: 154.08 MB
      wall  mean: 0.000058s  medi: 0.000058s  min: 0.000055s  max: 0.000062s  std: 0.000004s
      CPU   mean: 0.000059s  medi: 0.000059s  min: 0.000056s  max: 0.000062s  std: 0.000003s


Modules
-------

A module is a configurable building block which can be attached to a pipeline.
It has a ``process()`` method, which is called every time with the current
data ("blob") in the pipeline cycle. This piece of data can be analysed,
manipulated and finally returned to allow the handover to the next module
in the pipeline system.

Instance variables can be initialised within the ``configure()`` method.
User defined parameters are accessible via ``self.get()`` or ``self.required()``
calls inside the ``configure()`` method.
Both of them return the passed value or in case of ``self.get()``, the ``default=None``
value.
This allows an easy way to define default values as seen in the example below.

.. literalinclude:: ../examples/nogallery/module_workflow.py
   :pyobject: Foo
   :emphasize-lines: 4-6
   :linenos:

To override the default parameters, the desired values can be set when
attaching the module to the pipeline. Always use the class itself, since
the ``attach()`` method of the pipeline will care about the initialisation::

    pipe.attach(Foo, bar='dummybar', baz=69)

Module can optionally implement the following methods::

    configure()
    prepare()
    process()
    finish()

Pumps / Sinks
-------------

The pump and sink are special kinds of ``Module`` and are usually the
first and last ones to be attached to a pipeline. They are responsible
for reading and writing data to/from files, or streams from socket
connections in each iteration, continuously.
Note that they still simply derive from ``Module``.

``configure()`` method should be used to set up the file or socket
handler and the ``finish()`` has to close them. The actual data is
passed via the ``process()`` method. A data chunk is internally called
``Blob`` (a fancier ``dict``) and contains whatever the pump extracts
in each iteration. It can be a single event or the contents of a whole
file.

To end the data pumping, the pump has to raise a ``StopIteration``
exception. One elegant way to implement this in Python is using a
generator.

The following example shows a very basic pump, which simply initialises
a list of dictionaries and "io" one blob after another on each
``process()`` call to the next module in the pipeline.


.. literalinclude:: ../examples/nogallery/module_workflow.py
   :pyobject: DummyPump
   :linenos:


Logging and Printing
--------------------

Every module inheriting from the ``Module`` class has a fancy logger and a
printer available to produce output which is unique (an actual colour code
is generated using a hash of the module name).

Inside any method of the module, use ``self.log`` to access the logger, which
comes with the usual functions like ``self.log.debug()``, ``self.log.info()``,
``self.log.warning()``, ``self.log.error()`` or ``self.log.critical()``.

The ``self.cprint`` function can be used to print messages which are colour
coded with the same colours used for the logger.


Configuring the Pipeline using Configuration Files
--------------------------------------------------

The pipeline and all the attached modules can be configured by a TOML formatted
file, sitting at the current working directory (where the initial script is
invoked to launch the pipeline). The default filename is ``pipeline.toml``
but a different filename can be chosen when creating the ``Pipeline`` instance
using ``Pipeline(configfile='your_desired_filename.toml')``.

TOML does not support variables, but a section called ``[VARIABLES]`` can be
defined and those will be substituted when used in other (module) sections.

Here is an example of the file:

.. literalinclude:: ../examples/nogallery/pipeline.toml
   :linenos:
