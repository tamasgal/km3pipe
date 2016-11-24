Table / SciPy Workflow
======================

.. contents:: :local:

This page only gives a brief overview of the SciPy family.
If you want get started on Python for Science, I *highly* recommend the
SciPy Lecture Notes at `scipy-lectures.org <http://www.scipy-lectures.org/>`_.

Install ``pandas``, a.k.a. *the* data analysis library for python:
`pandas.pydata.org <http://http://pandas.pydata.org/>`_. Also, for some 
lower-level hdf5 file access, you might want to install ``tables``: `pytables.org <www.pytables.org>`_.::

    $ pip install pandas tables

Read A HDF5 File
----------------


Write a Pandas DataFrame to HDF5
--------------------------------

Use either the km3pipe HDF5 writers: ``df_to_h5`` (for pandas DataFrames), 
or ``write_table`` (for structured numpy arrays):

.. code-block:: python

    from km3pipe.io import df_to_h5, write_table

    df_to_h5(my_dataframe, 'bar.h5', tabname='MyTableName', where='/misc')

    # or, for numpy
    write_table('foo.h5', '/misc', array=my_numpy_array)

Or (for numpy arrays), if you are more experienced, use the pytables/h5py libraries directly.

Do **not** use ``pandas.HDFStore.write``, it stores the data in a weird format.
