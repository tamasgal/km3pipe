FROM debian:buster-slim
 MAINTAINER Tamas Gal <tgal@km3net.de>

 ENV INSTALL_DIR /km3pipe

 RUN apt-get update
 RUN apt-get install -y -qq git gnupg1 make python3 python3-pip wget
 RUN python3 -m pip install --upgrade pip setuptools wheel

 # LLVM 10 for Numba
 RUN echo "deb http://apt.llvm.org/buster/ llvm-toolchain-buster-10 main" > /etc/apt/sources.list.d/llvm10.list && \
     wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
     apt-get update && \
     apt-get install -y -qq llvm-10
 ENV LLVM_CONFIG /usr/lib/llvm-10/bin/llvm-config

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
