FROM python:3.12.7-bookworm
LABEL maintainer="Tamas Gal <tgal@km3net.de>"

 ENV INSTALL_DIR /km3pipe

 RUN apt-get update
 RUN apt-get install -y -qq git gnupg1 make wget llvm
 RUN python3 -m pip install --upgrade pip setuptools wheel

 ADD . $INSTALL_DIR
 RUN cd $INSTALL_DIR && \
     python3 -m pip install numpy && \
     python3 -m pip install . ".[dev]" ".[extras]"

 # Clean up
 RUN cd / && rm -rf $INSTALL_DIR
 RUN apt-get -y clean && apt-get autoclean && rm -rf /var/cache

 # 1337
 RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/motd' \
    >> /etc/bash.bashrc \
    ; echo "\\n\
    _/                        _/_/_/              _/                  \n\
   _/  _/    _/_/_/  _/_/          _/  _/_/_/        _/_/_/      _/_/ \n\
  _/_/      _/    _/    _/    _/_/    _/    _/  _/  _/    _/  _/_/_/_/\n\
 _/  _/    _/    _/    _/        _/  _/    _/  _/  _/    _/  _/      \n\
_/    _/  _/    _/    _/  _/_/_/    _/_/_/    _/  _/_/_/      _/_/_/ \n\
                                   _/            _/                  \n\
                                  _/            _/                   \n\
\n$(km3pipe --version)\n\
(c) Tamas Gal, Moritz Lotze and the KM3NeT Collaboration\n"\
    > /etc/motd
