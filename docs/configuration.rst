Configuration
=============

KM3Pipe can read frequently used information (like DB credentials, API tokens,
etc.) from a configuration file, which is expected to be `~/.km3net`.

Here is an example configuration::

    [General]
    check_for_updates=no

    [DB]
    username=fooman
    password=god

    [Slack]
    token=xoxp-2355837568-2397897846-8945924372-395f023485
