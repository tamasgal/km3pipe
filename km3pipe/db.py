# coding=utf-8
# Filename: db.py
# pylint: disable=locally-disabled
"""
Database utilities.

"""
from __future__ import division, absolute_import, print_function

from datetime import datetime
import ssl
import urllib
from urllib2 import (Request, build_opener, HTTPCookieProcessor, HTTPHandler)
import cookielib
import json
import sys

import pandas as pd

from km3pipe.tools import Timer
from km3pipe.config import Config
from km3pipe.logger import logging

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

__author__ = 'tamasgal'

log = logging.getLogger(__name__)  # pylint: disable=C0103


# Ignore invalid certificate error
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    log.warn("Your SSL support is outdate. "
             "Please update your Python installation!")

LOGIN_URL = 'https://km3netdbweb.in2p3.fr/home.htm'
BASE_URL = 'https://km3netdbweb.in2p3.fr'


class DBManager(object):
    """A wrapper for the KM3NeT Web DB"""
    def __init__(self, username=None, password=None):
        "Create database connection"
        self.cookies = []
        self._parameters = None
        self._doms = None
        self._detectors = None
        self._opener = None
        if username is None:
            config = Config()
            username, password = config.db_credentials
        self.login(username, password)

    def datalog(self, parameter, run, maxrun=None, detid='D_ARCA001'):
        "Retrieve datalogs for given parameter, run(s) and detector"
        parameter = parameter.lower()
        if maxrun is None:
            maxrun = run
        with Timer('Database lookup'):
            return self._datalog(parameter, run, maxrun, detid)

    def _datalog(self, parameter, run, maxrun, detid):
        "Extract data from database"
        values = {'parameter_name': parameter,
                  'minrun': run,
                  'maxrun': maxrun,
                  'detid': detid,
                  }
        data = urllib.urlencode(values)
        content = self._get_content('streamds/datalognumbers.txt?' + data)
        if content.startswith('ERROR'):
            log.error(content)
            return None
        try:
            dataframe = pd.read_csv(StringIO(content), sep="\t")
        except ValueError:
            log.warning("Empty dataset")  # ...probably. Waiting for more info
            return pd.DataFrame()
        else:
            def convert_data(timestamp):
                return datetime.fromtimestamp(float(timestamp) / 1e3)
            dataframe['DATETIME'] = dataframe['UNIXTIME'].apply(convert_data)
            convert_unit = self.parameters.get_converter(parameter)
            dataframe['VALUE'] = dataframe['DATA_VALUE'].apply(convert_unit)
            dataframe.unit = self.parameters.unit(parameter)
            return dataframe

    def run_table(self, detid='D_ARCA001'):
        url = 'streamds/runs.txt?detid={0}'.format(detid)
        content = self._get_content(url)
        try:
            dataframe = pd.read_csv(StringIO(content), sep="\t")
        except ValueError:
            log.warning("Empty dataset")
            return None
        else:
            def convert_data(timestamp):
                return datetime.fromtimestamp(float(timestamp) / 1e3)
            converted = dataframe['UNIXSTARTTIME'].apply(convert_data)
            dataframe['DATETIME'] = converted
            return dataframe

    @property
    def detectors(self):
        if not self._detectors:
            self._detectors = self._get_detectors()
        return self._detectors

    def _get_detectors(self):
        content = self._get_content('streamds/detectors.txt')
        try:
            dataframe = pd.read_csv(StringIO(content), sep="\t")
        except ValueError:
            log.warning("Empty dataset")
            return pd.DataFrame()
        else:
            return dataframe

    @property
    def parameters(self):
        "Return the parameters container for quick access to their details"
        if self._parameters is None:
            self._load_parameters()
        return self._parameters

    def _load_parameters(self):
        "Retrieve a list of available parameters from the database"
        parameters = self._get_json('allparam/s')
        data = {}
        for parameter in parameters:
            data[parameter['Name'].lower()] = parameter
        self._parameters = ParametersContainer(data)

    @property
    def doms(self):
        if self._doms is None:
            self._load_doms()
        return self._doms

    def _load_doms(self):
        "Retrieve DOM information from the database"
        doms = self._get_json('domclbupiid/s')
        self._doms = DOMContainer(doms)

    def _get_json(self, url):
        "Get JSON-type content"
        content = self._get_content('jsonds/' + url)
        json_content = json.loads(content)
        if json_content['Result'] != 'OK':
            raise ValueError('Error while retrieving the parameter list.')
        return json_content['Data']

    def _get_content(self, url):
        "Get HTML content"
        f = self.opener.open(BASE_URL + '/' + url)
        content = f.read()
        return content

    @property
    def opener(self):
        "A reusable connection manager"
        if self._opener is None:
            opener = build_opener()
            for cookie in self.cookies:
                cookie_str = cookie.name + '=' + cookie.value
                opener.addheaders.append(('Cookie', cookie_str))
            self._opener = opener
        return self._opener

    def login(self, username, password):
        "Login to the databse and store cookies for upcoming requests."
        cj = cookielib.CookieJar()
        opener = build_opener(HTTPCookieProcessor(cj), HTTPHandler())
        values = {'usr': username, 'pwd': password}
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
        self._converters = {}

    @property
    def names(self):
        "A list of parameter names"
        return self._parameters.keys()

    def get_parameter(self, parameter):
        "Return a dict of given parameter"
        return self._parameters[parameter]

    def get_converter(self, parameter):
        """Generate unit conversion function for given parameter"""
        if parameter not in self._converters:
            param = self.get_parameter(parameter)
            try:
                scale = float(param['Scale'])
            except KeyError:
                scale = 1

            def convert(value):
                # easy_scale = float(param['EasyScale'])
                # easy_scale_multiplier = float(param['EasyScaleMultiplier'])
                return value * scale

            return convert

    def unit(self, parameter):
        "Get the unit for given parameter"
        return self._parameters[parameter.lower()]['Unit']


class DOMContainer(object):
    """Provides easy access to DOM parameters"""
    def __init__(self, doms):
        self._doms = doms
