User Guide
==========

KM3Pipe has two main workflows: An Event-by-event processing approach, 
and a 2D-table/array approach. 

The event-by-event approach is handled by
the km3pipe *Pipeline*, where you chain together different modules for 
processing events (e.g. Hit cleaning, reconstruction), and then pump events --
also referred to as *blobs* -- down the pipeline.

The table-based approach is using the python libraries ``numpy`` and ``pandas``
from the SciPy ecosystem, which are fast and easy tools for scientific 
computation and data analysis, based on a clean tabular data format.

The preferred on-disk file format is :doc:`hdf5`, although km3pipe can read
all ROOT/Text based km3net file format like Jpp, Aanet, Evt, etc.

.. toctree:: 

    pipeline
    table
    hdf5
    cmd
    db
