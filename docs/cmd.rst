Command Line Interface
======================

If you've installed KM3Pipe via `pip`, you have access to some useful
command line utilities out of the box.
Most of the commands have to be prefixed with `km3pipe` to avoid possible
nameclashes and also for an improved overview.
You can for example simply run `km3pipe -h` in your shell to see all available
commands::

    KM3Pipe command line utility.

    Usage:
        km3pipe test
        km3pipe update [GIT_BRANCH]
        km3pipe tohdf5 [-n EVENTS] -i FILE -o FILE
        km3pipe aatohdf5 [-n EVENTS] -i FILE -o FILE
        km3pipe jpptohdf5 [-n EVENTS] -i FILE -o FILE
        km3pipe evttohdf5 [-n EVENTS] -i FILE -o FILE
        km3pipe hdf2root -i FILE [-o FILE]
        km3pipe runtable [-n RUNS] [-s REGEX] DET_ID
        km3pipe (-h | --help)
        km3pipe --version

    Options:
        -h --help       Show this screen.
        -i FILE         Input file.
        -o FILE         Output file.
        -n EVENTS/RUNS  Number of events/runs.
        -s REGEX        Regular expression to filter the runsetup name/id.
        DET_ID          Detector ID (eg. D_ARCA001).
        GIT_BRANCH          Git branch to pull (eg. develop).

`update`
~~~~~~~~

The command `km3pipe update [GIT_BRANCH]` should be used to (once installed)
get latest version of KM3Pipe. If no git branch is specified, it will pull
the master branch, which always holds the stable releases.

If you want to try the newest features, pull the develop branch via
`km3pipe update develop`. This is 99.9% stable, since we always do our
experiments in `feature/x` branches. However, we might break it sometimes.
Have a look at our git repository to see what we're working on if you're
interested.

`runtable`
~~~~~~~~~~

To get a list of runs taken with one of the KM3NeT detectors, you can use
the `runtable` command.

The following command pulls the last 10 runs which matches the regular
expression `PHYS`. In other words, you'll get a list of physics runs::

    km3pipe runtable -n 10 -s PHYS 14

An example output is::

            RUN	UNIXSTARTTIME	STARTTIME_DEFINED	RUNSETUPID	RUNSETUPNAME	T0_CALIBSETID	DATETIME
    848	3611	1465506000553	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-09 21:00:00.553000+00:00
    849	3612	1465506060554	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-09 21:01:00.554000+00:00
    850	3613	1465509600606	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-09 22:00:00.606000+00:00
    851	3614	1465509660607	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-09 22:01:00.607000+00:00
    852	3615	1465520400799	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 01:00:00.799000+00:00
    853	3616	1465520460800	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 01:01:00.800000+00:00
    854	3617	1465531200966	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 04:00:00.966000+00:00
    855	3618	1465531260967	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 04:01:00.967000+00:00
    856	3619	1465542000119	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 07:00:00.119000+00:00
    857	3620	1465542060119	Y	A01466427	PHYS.1606v1-TMP.HV-SFP.Power-XTRA.700ns		2016-06-10 07:01:00.119000+00:00



