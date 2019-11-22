Local DB
========

.. contents:: :local:

The ``km3modules.LocalDBService`` can be used to create and fill locally stored
(file-based) sqlite3 databases. You can either use this class standalone or
attach to a pipeline and use it as a service.

The service interface provides the following functions, which are exposed
to any attached module via::

    self.services["create_table"]()   # Use this to create SQL tables
    self.services["table_exists"]()   # To check if a table is already present
    self.services["insert_row"]()     # Insert data into a table

Standalone
~~~~~~~~~~

Here is an example how to create a table ``foo`` with two columns ``a`` and
``b`` with the types ``INT`` and ``TEXT`` and fill some rows::

    >>> import km3modules as km
    >>> dbs = km.common.LocalDBService(filename="db.sqlite")
    ++ km3modules.common.LocalDBService.LocalDBService: 2.6.0
    >>> dbs.table_exists("foo")
    False
    >>> dbs.create_table("foo", ["a", "b"], ["INT", "TEXT"])
    >>> dbs.insert_row("foo", ["a", "b"], (23, "hi mom!"))
    INSERT INTO foo (a, b) VALUES ('23','hi mom!')
    >>> dbs.insert_row("foo", ["a", "b"], (42, "Monty Python"))
    INSERT INTO foo (a, b) VALUES ('42','Monty Python')
    >>> dbs.insert_row("foo", ["a"], (5,))
    INSERT INTO foo (a) VALUES ('5')

Data retrieval (TODO)

As a Service
~~~~~~~~~~~~

...
