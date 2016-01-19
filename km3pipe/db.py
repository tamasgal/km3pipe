# coding=utf-8
# Filename: db.py
# pylint: disable=locally-disabled
"""
Database utilities.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from datetime import datetime
import ConfigParser, os
import ssl
import urllib
from urllib2 import (Request, build_opener, HTTPCookieProcessor, HTTPHandler)
import cookielib
import json
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import pandas as pd

from km3pipe.config import Config

import logging
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


# Ignore invalid certificate error
ssl._create_default_https_context = ssl._create_unverified_context

LOGIN_URL='https://km3netdbweb.in2p3.fr/home.htm'
BASE_URL='https://km3netdbweb.in2p3.fr'


class DBManager(object):
    def __init__(self, username=None, password=None):
        self.cookies = []
        self._parameters = None
        self._opener = None
        if username is None:
            config = Config()
            username, password = config.db_credentials
        self.login(username, password)

    def datalog(self, parameter, run, maxrun=None, detid='D_DU2NAPO'):
        if maxrun is None:
            maxrun = run
        values = { 'parameter_name': parameter.lower(),
                   'minrun': run,
                   'maxrun': maxrun,
                   'detid': detid,
                   }
        data = urllib.urlencode(values)
        content = self._get_content('streamds/datalognumbers.txt?' + data)
        try:
            #dataframe = pd.read_csv(StringIO(content), sep="\t",
            #                        parse_dates=['UNIXTIME'],
            #                        date_parser=convert)
            dataframe = pd.read_csv(StringIO(content), sep="\t")
        except ValueError:
            log.warning("Empty dataset")
            return None
        else:
            convert = lambda x: datetime.fromtimestamp(float(x) / 1e3)
            dataframe['DATETIME'] = dataframe['UNIXTIME'].apply(convert)
            return dataframe

    @property
    def parameters(self):
        if self._parameters is None:
            self._load_parameters()
        return self._parameters

    def _load_parameters(self):
        parameters = self._get_json('allparam/s')
        if parameters['Result'] != 'OK':
            raise ValueError('Error while retrieving the parameter list.')
        data = {}
        for parameter in parameters['Data']:
            data[parameter['Name'].lower()] = parameter
        self._parameters = ParametersContainer(data)

    def _get_json(self, url):
        content = self._get_content('jsonds/' + url)
        return json.loads(content)

    def _get_content(self, url):
        f = self.opener.open(BASE_URL + '/' + url)
        content = f.read()
        return content

    @property
    def opener(self):
        if self._opener is None:
            opener = build_opener()
            for cookie in self.cookies:
                cookie_str = cookie.name + '=' + cookie.value
                opener.addheaders.append(('Cookie', cookie_str))
            self._opener = opener
        return self._opener

    def login(self, username, password):
        cj = cookielib.CookieJar()
        opener = build_opener(HTTPCookieProcessor(cj), HTTPHandler())
        values = { 'usr': username,'pwd': password }
        data = urllib.urlencode(values)
        req = Request(LOGIN_URL, data)
        f = opener.open(req)
        html = f.read()
        if 'Bad username or password' in html:
            log.error("Bad username or password!")
        self.cookies = cj


class ParametersContainer(object):
    """Provides easy access to parameters"""
    def __init__(self, parameters):
        self._parameters = parameters

    @property
    def names(self):
        return self._parameters.keys()

    def unit(self, parameter):
        return self._parameters[parameter.lower()]['Unit']
