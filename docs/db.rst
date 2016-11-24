DBManager
=========

.. contents:: :local:

The ``DBManager`` class provides an easy access to data stored in the KM3NeT
Oracle database and takes care of the whole authentication procedure. It uses
an in-memory cache for the fetched data to reduce network traffic and I/O.
All you need to do is create an instance of the ``DBManager`` class.

Dataformats
~~~~~~~~~~~
The database web API uses three different formats: ASCII (CSV like), JSON and
XML. All the ASCII output is parsed in ``DBManager`` and converted to
pandas_ ``DataFrames``, which is the de facto standard for representing large
data in Python_.
Great benefits of using pandas_ is its well written documentation, a huge
number of tutorials/books, tens of thousands of questions and answers on
stackoverflow_ and of course the nice and active scientific community
behind it.
The JSON output of the database web API is wrapped in Python classes to make
your life easier.

.. _Python: http://www.python.org
.. _pandas: http://pandas.pydata.org
.. _stackoverflow: http://www.stackoverflow.com


DB Authentication
~~~~~~~~~~~~~~~~~
There are two ways to create a database connection, either using your KM3NeT
database username and password, or via a special permanent session cookie
which can be requested after a successful login.

If you create a ``DBManager`` instance without any initialisation arguments,
it will ask you for your username and password and upon successful login
it gives you the option to request a permanent session cookie and stores it
in your KM3Pipe configuration file under ``~/.km3net``::

    >>> import km3pipe as kp
    >>> db = kp.db.DBManager()
    Please enter your KM3NeT DB username: tgal
    Password:
    Request permanent session? (y/n)y

Your ``~/.km3net`` configuration file should now look something like this::

    [DB]
    session_cookie = sid=_username_12.34.56.78_504c293fac5b4ddbba2cfc3dc33eaadc

If you ever encounter any issues with the database communication, try deleting
the ``session_cookie`` line in the configuration file and request a new one
as described above.

Important note: due to security reasons, the ``~/.km3net`` configuration file
should not be readable to others. KM3Pipe will warn you and will also display
the command to set the appropriate permission::

    chmod 600 ~/.km3net

Retrieving the list of detectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The registered detectors can be access via::

    >>> db.detectors
              OID  SERIALNUMBER LOCATIONID       CITY
    0   D_DU1CPPM             2  A00070004  Marseille
    1   A00350276             3  A00070003     Napoli
    2   A00350280             4  A00070005    Bologna
    3   D_DU2NAPO             5  A00070003     Napoli
    4   D_TESTDET             6  A00070002   Fisciano
    5   D_ARCA001             7  A00073795      Italy
    6   FR_INFRAS             8  A00073796     France
    7   D_DU003NA             9  A00070003     Napoli
    8   D_DU004NA            12  A00070003     Napoli
    9   D_DU001MA            13  A00070004  Marseille
    10  D_ARCA003            14  A00073795      Italy
    11  A01495728            21  A00074177  Amsterdam
    12  A01495762            22  A00074177  Amsterdam
    13  D_BCI0001            23  A00070002   Fisciano

The data you get is a ``pandas.DataFrame`` instance, which represents itself
as an ASCII table.

To demonstrate the table like ``DataFrame`` structure, see the following tiny
example on how to create a list of the cities or look for the ``OID``
given a detector serialnumber::

    >>> dts = db.detectors
    >>> dts.CITY.drop_duplicates().sort_values()
    11    Amsterdam
    2       Bologna
    4      Fisciano
    6        France
    5         Italy
    0     Marseille
    1        Napoli
    >>> dts[dts.SERIALNUMBER == 14].OID
    10    D_ARCA003
    Name: OID, dtype: object

Don't be confused about the left column, those are the actual indices of the
rows in the original pandas ``DataFrame``.


Fun with DOMs
~~~~~~~~~~~~~
To retrieve information about DOMs, the ``DBManager`` provides a handy
``DOMContainer`` class, which can be access via::

    >>> db.doms
    <km3pipe.db.DOMContainer object at 0x110daea10>

You can take a look at the docstring of the class using Pythons ``help``
function::

    >>> help(db.doms)
    class DOMContainer(__builtin__.object)
     |  Provides easy access to DOM parameters stored in the DB.
     |
     |  Methods defined here:
     |
     |  __init__(self, doms)
     |
     |  clbupi2domid(self, clb_upi, det_id)
     |      Return DOM ID for given CLB UPI and detector
     |
     |  clbupi2floor(self, clb_upi, det_id)
     |      Return Floor ID for given CLB UPI and detector
     |
     |  domid2floor(self, dom_id, det_id)
     |      Return Floor ID for given DOM ID and detector
     |
     |  ids(self, det_id)
     |      Return a list of DOM IDs for given detector
     |
     |  via_clb_upi(self, clb_upi)
     |      return DOM for given CLB UPI
     |
     |  via_dom_id(self, dom_id)
     |      Return DOM for given dom_id
     |
     |  via_omkey(self, omkey, det_id)
     |      Return DOM for given OMkey (DU, floor)

The most important methods are probablly ``via_clb_upi``, ``via_dom_id`` and
``via_omkey``. All of them will return an instance of ``DOM`` which is
basically a struct, holding the usual DOM information.
The ``via_omkey`` method takes a tuple ``(DU, floor)`` and also requires the
detector OID.
Here are some examples how to use these methods::

    >>> a_dom = db.doms.via_omkey((2, 16), "D_ARCA003")
    >>> a_dom
    DU2-DOM16 - DOM ID: 809548782
       DOM UPI: 3.4/CH25H/1.60
       CLB UPI: 3.4.3.2/V2-2-1/2.594
       DET OID: D_ARCA003

    >>> print(a_dom)
    DU2-DOM16
    >>> a_dom.clb_upi
    u'3.4.3.2/V2-2-1/2.594'
    >>> a_dom.floor
    16
    >>> a_dom.du
    2

    >>> another_dom = db.doms.via_clb_upi("3.4.3.2/V2-2-1/2.296")
    >>> print(another_dom)
    DU2-DOM9
    >>> another_dom
    DU2-DOM9 - DOM ID: 808951763
       DOM UPI: 3.4/CH39H/1.53
       CLB UPI: 3.4.3.2/V2-2-1/2.296
       DET OID: D_ARCA003
