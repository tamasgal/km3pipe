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


Pumps
-----

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
list of dictionaries and "io" one blob after another on
each ``process()`` call to the next module in the pipeline.


.. literalinclude:: ../examples/module_workflow.py
   :pyobject: DummyPump
   :linenos:

Data Structures
---------------

This section describes the basic data structures which a **pump**
provides via the **blob** dictionary. The pump is responsible to parse
the data and create a **blob** (a simple Python dictionary) for each
event in the file. When processing a data file with KM3Pipe, a module
chain is being utilised to cycle through the events. Each module within
the chain recieves the original, unaltered data from the pump and
further also additional information created by the preceeding modules.

Hits
~~~~

There are two kinds of basic hit types: a **raw hit** representing either an
actual hit measured by the detector hardware or a calibrated MC hit which
does not contain MC information anyomre, and a **MC hit**, which
was created by a Monte Carlo simulation. The dictonary key naming
conventions for raw hits and MC hits are the following:

+---------------+------------+------------------------+
| information   | dict key   | container type         |
+===============+============+========================+
| Raw Hits      | Hits       | HitSeries (list-like)  |
+---------------+------------+------------------------+
| MC Hits       | MCHits     | HitSeries (list-like)  |
+---------------+------------+------------------------+

Both hit types have attributes which can be accessed through the
following getters:

+---------------------+--------------+-----------+-----------+----------+
| information         | getter       | type      | raw hit   | MC hit   |
+=====================+==============+===========+===========+==========+
| hit id              | .id          | numeric   | X         | X        |
+---------------------+--------------+-----------+-----------+----------+
| hit time            | .time        | numeric   | X         | X        |
+---------------------+--------------+-----------+-----------+----------+
| time over threshold | .tot         | numeric   | X         |          |
+---------------------+--------------+-----------+-----------+----------+
| PMT id              | .pmt_id      | numeric   | X         | X        |
+---------------------+--------------+-----------+-----------+----------+
| Channel id          | .channel_id  | numeric   | X         | X        |
+---------------------+--------------+-----------+-----------+----------+
| trigger information | ...          | ...       | X         |          |
+---------------------+--------------+-----------+-----------+----------+

to be continued...

