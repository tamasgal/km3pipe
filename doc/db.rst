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

You can also add your username and DB password to the ``~/.km3net`` config 
file like this::

    [DB]
    session_cookie = sid=...
    username = MYNAME
    password = INTERNALPASS

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


CLBs
~~~~

The ``CLBMap`` class is a convenient tool to check the CLB parameters like
UPI, floor, DU or just to find out a base for a given DU::

    >>> import km3pipe as kp
    >>> clbmap = kp.db.CLBMap("D_ORCA003")  # use the det OID
    >>> clbmap.base(1)
    CLB(
        det_oid='D_ORCA003',
        floor=0,
        du=1,
        serial_number=267,
        upi='3.4.3.2/V2-2-1/2.267',
        dom_id=808476701
    )
    >>> clbmap.upi['3.4.3.2/V2-2-1/2.267'].dom_id
    808476701
    >>> clbmap.dom_id[808959411].floor
    5

Fun with DOMs
~~~~~~~~~~~~~
**Important note: the following method will be deprecated soon and replaced
by the `CLBMap` as described in the previous subsection.**

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
    '3.4.3.2/V2-2-1/2.594'
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


Datalogs
~~~~~~~~
This is probably the most interesting part of the database. The datalogs
is a meta table which provides access to hundreds of different parameter types.

Parameters
^^^^^^^^^^

The available parameters can be inspected via the ``ParametersContainer`` class
which is -- just like the ``DOMContainer`` -- automatically instantiated and
accessible as an attribute of the ``DBManager``::

    >>> db.parameters
    <km3pipe.db.ParametersContainer object at 0x110d22250>

A quick peek on ``help(db.parameters)`` reveals a few methods and attributes::

    >>> help(db.parameters)
    class ParametersContainer(__builtin__.object)
     |  Provides easy access to parameters
     |
     |  Methods defined here:
     |
     |  __init__(self, parameters)
     |
     |  get_converter(self, parameter)
     |      Generate unit conversion function for given parameter
     |
     |  get_parameter(self, parameter)
     |      Return a dict for given parameter
     |
     |  unit(self, parameter)
     |      Get the unit for given parameter
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  names
     |      A list of parameter names

The ``names`` attribute gives you a list of available parameters::

    >>> len(db.parameters.names)
    277
    >>> db.parameters.names[:5]
    ['led_model', 'pmt_serialnumber', 'bps_breaker', 'humid',
    'pwr_meas[9] power_measurement_12v_lvl']

The above example shows the first 5 parameters out of 277 entries.
If you see a number enclosed by brackets in a parameter name, like
``"pwr_meas[9] power_measurement_12v_lvl"`` in the list above, it means that
``"pwr_meas"`` is a parameter-array and the value at index ``9`` is aliased to
``power_measurement_12v_lvl``. The latter name should be used if you want
to retrieve the corresponding data from the DB.

Parameter Units and Value Conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ParametersContainer`` has three methods to access information about a
given parameter.
The ``get_converter()`` method returns a function to be used to convert
the raw values stored for a given parameter to match the target unit, which
is returned by the ``unit()`` method::

    >>> humid_converter = db.parameters.get_converter("humid")
    >>> humid_converter(987)
    9.870000000000001
    >>> db.parameters.unit("humid")
    '%'

Retrieving Parameter Data
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``datalog`` method provides an easy way to retrieve data for a given
detector and run or range of runs. It returns a pandas ``DataFrame`` instance::

    >>> humid = db.datalog("humid", run=4780, det_id="D_ARCA003")
    Database lookup took 3.931s (CPU 0.192s).
    >>> type(humid)
    <class 'pandas.core.frame.DataFrame'>

The ``head()`` and ``tail()`` methods can be used to get the first or last
rows::

    >>> humid.head(3)
        RUN       UNIXTIME           SOURCE_NAME PARAMETER_NAME  DATA_VALUE  \
    0  4780  1478735722766  3.4.3.2/V2-2-1/2.138          humid        3694
    1  4780  1478735732768  3.4.3.2/V2-2-1/2.138          humid        3694
    2  4780  1478735742766  3.4.3.2/V2-2-1/2.138          humid        3694

                              DATETIME  VALUE
    0 2016-11-09 23:55:22.766000+00:00  36.94
    1 2016-11-09 23:55:32.768000+00:00  36.94
    2 2016-11-09 23:55:42.766000+00:00  36.94

The ``DATA_VALUE`` is the column which holds the recorded data
(the "raw values"). The ``VALUE`` column is automatically added by the
``DBManager`` -- if the parameter has a valid unit and conversion score entry in
the database -- by applying the above mentioned ``get_converter()`` method
on the ``DATA_VALUE`` column.
If the data contains a ``UNIXTIME`` column, a ``DATETIME`` field will be added
too, which allows using all the magical date filtering methods.


StreamDS
~~~~~~~~

You already learned how to use the ``DBManager`` to connect to the database
and access information. The ``StreamDS`` class is a specific helper, which
connects to the StreamDS_ (Stream Data Service) of the KM3NeT database web
server interface. The StreamDS is used to retrieve large datasets which could
possibly reach and exceed GB size.

.. _StreamDS: http://wiki.km3net.physik.uni-erlangen.de/index.php/Database/Stream_Data_Service

``StreamDS`` uses the ``DBManager`` to connect to the database and you
instantiate the same way::

    >>> import km3pipe as kp
    >>> sds = kp.db.StreamDS()
    Please enter your KM3NeT DB username: tgal
    Password:
    Request permanent session? (y/n)y

Notice that you won't be asked for the password or session if you already
put your credentials into your ``~/.km3net`` configuration or created a
permanent session before (and your IP has not changed since then).

If you type ``sds.`` and press ``<TAB>``, you will see a list of available
methods and getters for all available streams. The methods are generated
dynamically, so it is always up to date with the latest web API::

    >>> sds.
    sds.ahrs(                        sds.pmt_available_hvtuned_sets(
    sds.clbmap(                      sds.pmt_best_hv_settings(
    sds.clbmon(                      sds.pmt_hv_run_settings(
    sds.clbmondomid(                 sds.pmt_hv_settings(
    sds.clbmonpos(                   sds.pmt_hv_tuning_settings(
    sds.clbmonupi(                   sds.pmtdarkbox(
    sds.datalogevents(               sds.print_streams(
    sds.datalognumbers(              sds.runs(
    sds.datalogstrings(              sds.runsummarynumbers(
    sds.detcalibrations(             sds.streams
    sds.detectors(                   sds.t0(
    sds.dmvars(                      sds.t0sets(
    sds.get(                         sds.toa(
    sds.integration(                 sds.toashort(
    sds.jobs(                        sds.upi(
    sds.mandatory_selectors(         sds.vendorhv(
    sds.optional_selectors(          sds.vendorhvrunsetup(

To get a full list of available streams::

    >>> sds.streams
    ['detectors', 'runs', 'jobs', 'datalognumbers', 'datalogstrings',
     'datalogevents', 'vendorhv', 'vendorhvrunsetup', 't0sets', 't0',
     'ahrs', 'upi', 'pmtdarkbox', 'dmvars', 'detcalibrations',
     'pmt_hv_settings', 'pmt_hv_tuning_settings', 'pmt_hv_run_settings',
     'pmt_best_hv_settings', 'pmt_available_hvtuned_sets', 'integration',
     'clbmon', 'clbmonupi', 'clbmondomid', 'clbmonpos', 'clbmap', 'toa',
     'toashort', 'runsummarynumbers']

To print all streams including their selectors and data formats, use the
``sds.print_streams()`` function::

    >>> sds.print_streams()
    detectors
    Shows all the detectors, optionally selecting by site oid or city.
      available formats:   txt
      mandatory selectors: -
      optional selectors:  locationid,city

    runs
    Shows all runs for a detector (mandatory selection by detid or serialnumber). Optionally, a single run may be specified.
      available formats:   txt
      mandatory selectors: detid
      optional selectors:  run

    jobs
    Shows all detector run jobs for a detector within a minimum and maximum Unix time (all mandatory selections). Optionally, selections may consider priority, runsetupid, oid.
      available formats:   txt
      mandatory selectors: detid,unixmintime,unixmaxtime
      optional selectors:  priority,runsetupid,oid,localid
    ...
    ...
    ...

If you are using ``ipython`` (recommended), you can get a quick help if you
type for example ``sds.vendorhv?`` to see what the ``vendorhv`` stream does and
which selectors it needs (if you are using the plain ``python`` REPL,
type ``help(sds.vendorhv)`` instead. Also notice that some completion features
are only supported for Python 3.3+ (you should update to Python 3.6 anyways...)::

    >>> sds.vendorhv?
    Signature: sds.vendorhv(detid, *, pmtserial)
    Docstring: Shows vendor-suggested HV for a detector (mandatory selection by detid or serialnumber). Optionally, a single PMT may be specified.
    File:      ~/Dev/km3pipe/km3pipe/db.py
    Type:      function

As you can see, the ``Signature`` indicates that ``detid`` is mandatory and
the keyword(s) after the ``*`` are optional (in this case ``pmtserial``).


Let's retrieve some data::

    >>> sds.vendorhv(detid=14)
      DUID  FLOORID  CABLEPOS  PMTSERIAL  PMT_SUPPLY_VOLTAGE
      0        1        1         0       1838               -1010
      1        2        1         0        704               -1080
      2        3        1         0       5586               -1030
      3        2        1         1       6461                -990
      4        3        1         1       6483               -1100
      5        1        1         1       4944                -930

That's it. You always get a Pandas ``DataFrame`` back. Have fun!


The ``streamds`` CLI
~~~~~~~~~~~~~~~~~~~~
There is also a command line utility called ``streamds``, which can be used to
interact with the database directly from the shell::

    $ streamds --help
    Access the KM3NeT StreamDS DataBase service.

    Usage:
        streamds
        streamds list
        streamds upload CSV_FILE
        streamds info STREAM
        streamds get STREAM [PARAMETERS...]
        streamds (-h | --help)
        streamds --version

    Options:
        STREAM      Name of the stream.
        CSV_FILE    Whitespace separated data for the runsummary tables.
        PARAMETERS  List of parameters separated by space (e.g. detid=29).
        -h --help   Show this screen.


Uploading "runsummarynumbers"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the ``streamds upload CSV_FILE`` command to upload data to
the "runsummarynumbers" meta table of the KM3NeT database. Please discuss
in advance any new types of parameters with database experts and create a
wiki page which describes them in detail.

The required columns are ``run``, ``det_id`` and ``source``. The ``source`` 
column is a free string-type column. It is recommended to use the DOM ID if 
you have parameters which refer to DOMs. If you have a column which refers to 
the whole run, use the string ``"run"`` in the source column e.g. for a 
parameter which refers to a DU, you can set it to ``"du1"`` etc.

Here is an example of a CSV file::

    det_id     run     source     n_active_doms highest_rate
    D_ARCA001  523     whole_run  18            230042
    D_ARCA001  523     du1        3             123000
    D_ARCA001  524     whole_run  17            500023

Please note that the whole file will be rejected if there is
even a single row of data which is already present in the database.

