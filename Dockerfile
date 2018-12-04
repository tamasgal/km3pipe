FROM docker.km3net.de/base/python:3
 MAINTAINER Tamas Gal <tgal@km3net.de>

 ADD . /km3pipe
 RUN cd /km3pipe && make install
