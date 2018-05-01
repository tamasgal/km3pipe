How to Write Documentation
--------------------------

Build docs locally::
  
  cd km3pipe/doc
  make html


Docstrings, Docstrings
~~~~~~~~~~~~~~~~~~~~~~

We (ok 1+) really like the numpy docstring style! Find it defined at

https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

If you want to check if your code follows the pydocstyle, run::

  pydocstyle --convention=numpy FILENAME

or simply (checks the whole project)::

  make docstyle

How to Add Examples to the Gallery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

