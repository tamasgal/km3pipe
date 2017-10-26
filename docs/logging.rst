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

    import km3pipe as kp
    log = kp.logger.get("module.path")

where ``module.path`` is the Python import path of the module, like ``km3pipe.core``.

To set a desired logging level, use the keywords ``DEBUG``, ``INFO``, ``WARNING``,
``ERROR`` or ``CRITICAL``. For example::

    log.setLevel("DEBUG")

Creating your own Logger
------------------------

To create your own logger, use the same procedure as described above::

    import km3pipe as kp
    log = kp.logger.get("your.desired.logger.name")

After that, you can use it to log anywhere::

    log.debug("A debug message")
    log.info("An info message")
    log.warning("A warning")
    log.error("An error")
    log.critical("A critical think")

and set its debug level::

    log.setLevel("WARN")


Modifying log levels of existing modules
----------------------------------------

The following script shows how to access the logger of the ``km3pipe.core``
module and set its log level individually.

.. literalinclude:: ../examples/nogallery/logging_example.py
   :language: python
      :linenos:

This is the output if you change the log level of ``km3pipe.core`` to ``DEBUG``::

    INFO:km3pipe.core:Attaching module 'foo'
    DEBUG:km3pipe.core:Attaching as function module
    Pipeline and module initialisation took 0.002s (CPU 0.001s).
    INFO:km3pipe.core:Trapping CTRL+C and starting to drain.
    INFO:km3pipe.core:Now draining...
    DEBUG:km3pipe.core:Pumping blob #0
    DEBUG:km3pipe.core:Processing foo
    Module called
    DEBUG:km3pipe.core:Pumping blob #1
    DEBUG:km3pipe.core:Processing foo
    Module called
    DEBUG:km3pipe.core:Pumping blob #2
    DEBUG:km3pipe.core:Processing foo
    Module called
    DEBUG:km3pipe.core:Pumping blob #3
    DEBUG:km3pipe.core:Processing foo
    Module called
    DEBUG:km3pipe.core:Pumping blob #4
    DEBUG:km3pipe.core:Processing foo
    Module called
    INFO:km3pipe.core:Nothing left to pump through.
    INFO:km3pipe.core:Skipping function module foo
    ============================================================
    5 cycles drained in 0.005251s (CPU 0.003115s). Memory peak: 78.95 MB
      wall  mean: 0.000523s  medi: 0.000263s  min: 0.000168s  max: 0.001690s  std: 0.000585s
      CPU   mean: 0.000308s  medi: 0.000264s  min: 0.000170s  max: 0.000611s  std: 0.000157s
