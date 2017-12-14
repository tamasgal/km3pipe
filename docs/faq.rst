FAQ
===

.. contents:: :local:

This section describes some more obscure things relevant to day-to-day
data analysis, which are not really documented explicitly.

AAnet formats (``tohdf5 --aa-format=...``)
------------------------------------------

Ancient recolns (ORCA recolns):
``tohdf5 --aa-format=ancient_recolns --aa-lib=${LIAM}/sandbox/aa-recoLNSlowE-prod2016/bin/reco_v0r9_standalone.so``

Gandalf (orca MX and later):
``tohdf5 --aa-format=gandalf_new``


``AanetPump``/``tohdf5`` segfaults:
-----------------------------------

Might be the header:
``tohdf5 --skip-header --aa-format=gandalf_new``
