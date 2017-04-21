Logging
=======


.. contents:: :local:


Introduction
------------
KM3Pipe uses a module based logging system which can individually be configured to
provide information needed to understand the underlying mechanisms.

You can also easily create your own logger.


Accessing a Logger
------------------
To access a modules logger, you need to::

    from km3pipe.logger import logging
    log = logging.getLogger("module.path")

where ``module.path`` is the Python import path of the module, like ``km3pipe.core``.

To set a desired logging level, use the keywords ``DEBUG``, ``INFO``, ``WARNING``,
``ERROR`` or ``CRITICAL``. For example::

    log.setLevel("DEBUG")

Creating your own Logger
------------------------

To create your own logger, use the same procedure as described above::

    from km3pipe.logger import logging
    log = logging.getLogger("your.desired.logger.name")

After that, you can use it to log anywhere::

    log.debug("A debug message")
    log.info("An info message")
    log.warn("A warning")
    log.error("An error")
    log.critical("A critical think")

and set its debug level::

    log.setLevel("WARN")

