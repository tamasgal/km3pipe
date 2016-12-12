# coding=utf-8
# Filename: db.py
# pylint: disable=locally-disabled
"""
Database utilities.

"""
from __future__ import division, absolute_import, print_function

from datetime import datetime
import ssl
import sys
import json
import re
import pytz

try:
    import pandas as pd
except ImportError:
    print("The database utilities needs pandas: pip install pandas")

from km3pipe.tools import Timer
from km3pipe.config import Config
from km3pipe.logger import logging

try:
    input = raw_input
except NameError:
    pass

if sys.version_info[0] > 2:
    from urllib.parse import urlencode, unquote
    from urllib.request import (Request, build_opener, urlopen,
                                HTTPCookieProcessor, HTTPHandler)
    from urllib.error import URLError, HTTPError
    from io import StringIO
    from http.cookiejar import CookieJar
    from http.client import IncompleteRead
else:
    from urllib import urlencode, unquote
    from urllib2 import (Request, build_opener, urlopen,
                         HTTPCookieProcessor, HTTPHandler,
                         URLError, HTTPError)
    from StringIO import StringIO
    from cookielib import CookieJar
    from httplib import IncompleteRead

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103

UTC_TZ = pytz.timezone('UTC')


# Ignore invalid certificate error
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    log.debug("Your SSL support is outdated.\n"
              "Please update your Python installation!")

BASE_URL = 'https://km3netdbweb.in2p3.fr'


class DBManager(object):
    """A wrapper for the KM3NeT Web DB"""
    def __init__(self, username=None, password=None, url=None, temporary=False):
        "Create database connection"
        self._cookies = []
        self._parameters = None
        self._doms = None
        self._detectors = None
        self._opener = None
        self._temporary = temporary

        config = Config()

        if url is not None:
            self._db_url = url
        else:
            self._db_url = config.db_url or BASE_URL

        self._login_url = self._db_url + '/home.htm'

        if username is not None and password is not None:
            self.login(username, password)
        elif config.db_session_cookie not in (None, ''):
            self.restore_ression(config.db_session_cookie)
        else:
            username, password = config.db_credentials
            login_ok = self.login(username, password)
            if login_ok and not self._temporary and \
               input("Request permanent session? (y/n)") in 'yY':
                self.request_permanent_session(username, password)

    def datalog(self, parameter, run, maxrun=None, det_id='D_ARCA001'):
        "Retrieve datalogs for given parameter, run(s) and detector"
        parameter = parameter.lower()
        if maxrun is None:
            maxrun = run
        with Timer('Database lookup'):
            return self._datalog(parameter, run, maxrun, det_id)

    def _datalog(self, parameter, run, maxrun, det_id):
        "Extract data from database"
        values = {'parameter_name': parameter,
                  'minrun': run,
                  'maxrun': maxrun,
                  'detid': det_id,
                  }
        data = urlencode(values)
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
            self._add_datetime(dataframe)
            try:
                self._add_converted_units(dataframe, parameter)
            except KeyError:
                log.warn("Could not add converted units for {0}"
                         .format(parameter))
            return dataframe

    def run_table(self, det_id='D_ARCA001'):
        url = 'streamds/runs.txt?detid={0}'.format(det_id)
        content = self._get_content(url)
        try:
            dataframe = pd.read_csv(StringIO(content), sep="\t")
        except ValueError:
            log.warning("Empty dataset")
            return None
        else:
            self._add_datetime(dataframe, 'UNIXSTARTTIME')
            return dataframe

    def _add_datetime(self, dataframe, timestamp_key='UNIXTIME'):
        """Add an additional DATETIME column with standar datetime format.

        This currently manipulates the incoming DataFrame!
        """
        def convert_data(timestamp):
            return datetime.fromtimestamp(float(timestamp) / 1e3, UTC_TZ)
        try:
            converted = dataframe[timestamp_key].apply(convert_data)
            dataframe['DATETIME'] = converted
        except KeyError:
            log.warn("Could not add DATETIME column")

    def _add_converted_units(self, dataframe, parameter, key='VALUE'):
        """Add an additional DATA_VALUE column with converted VALUEs"""
        convert_unit = self.parameters.get_converter(parameter)
        try:
            dataframe[key] = dataframe['DATA_VALUE'].apply(convert_unit)
        except KeyError:
            log.warn("Missing 'VALUE': no unit conversion.")
        else:
            dataframe.unit = self.parameters.unit(parameter)

    @property
    def detectors(self):
        if self._detectors is None:
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

    def t0sets(self, det_id):
        content = self._get_content('streamds/t0sets.txt?detid={0}'
                                    .format(det_id))
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
        for parameter in parameters:  # There is a case-chaos in the DB
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

    def detx(self, det_id, t0set=None, calibration=None):
        """Retrieve the detector file for given detector id

        If t0set is given, append the calibration data.
        """
        url = 'detx/{0}?'.format(det_id)  # '?' since it's ignored if no args
        if t0set is not None:
            url += '&t0set=' + t0set
        if calibration is not None:
            url += '&calibrid=' + calibration

        detx = self._get_content(url)
        return detx

    def ahrs(self, run, maxrun=None, clbupi=None, det_id='D_ARCA001'):
        "Retrieve AHRS values for given run(s) (optionally CLBs) and detector"
        if maxrun is None:
            maxrun = run
        with Timer('Database lookup'):
            return self._ahrs(run, maxrun, clbupi, det_id)

    def _ahrs(self, run, maxrun, clbupi, det_id):
        values = {'minrun': run,
                  'maxrun': maxrun,
                  'detid': det_id,
                  }
        if clbupi is not None:
            values['clbupi'] = clbupi
        data = urlencode(values)
        content = self._get_content('streamds/ahrs.txt?' + data)
        if content.startswith('ERROR'):
            log.error(content)
            return None
        try:
            dataframe = pd.read_csv(StringIO(content), sep="\t")
        except ValueError:
            log.warning("Empty dataset")  # ...probably. Waiting for more info
            return pd.DataFrame()
        else:
            self._add_datetime(dataframe)
            return dataframe

    def _get_json(self, url):
        "Get JSON-type content"
        content = self._get_content('jsonds/' + url)
        try:
            json_content = json.loads(content.decode())
        except AttributeError:
            json_content = json.loads(content)
        if json_content['Comment']:
            log.warn(json_content['Comment'])
        if json_content['Result'] != 'OK':
            raise ValueError('Error while retrieving the parameter list.')
        return json_content['Data']

    def _get_content(self, url):
        "Get HTML content"
        target_url = self._db_url + '/' + unquote(url)  # .encode('utf-8'))
        log.debug("Opening '{0}'".format(target_url))
        try:
            f = self.opener.open(target_url)
        except HTTPError as e:
            log.error("HTTP error, your session may be expired.")
            log.error(e)
            if input("Request new permanent session and retry? (y/n)") in 'yY':
                self.request_permanent_session()
                return self._get_content(url)
            else:
                return None
        log.debug("Accessing '{0}'".format(target_url))
        try:
            content = f.read()
        except IncompleteRead as icread:
            log.critical("Incomplete data received from the DB, " +
                         "the data could be corrupted.")
            content = icread.partial
        log.debug("Got {0} bytes of data.".format(len(content)))
        return content.decode('utf-8')

    @property
    def opener(self):
        "A reusable connection manager"
        if self._opener is None:
            opener = build_opener()
            for cookie in self._cookies:
                cookie_str = cookie.name + '=' + cookie.value
                opener.addheaders.append(('Cookie', cookie_str))
            self._opener = opener
        return self._opener

    def request_sid_cookie(self, username, password):
        """Request cookie for permanent session token."""
        target_url = self._login_url + '?usr={0}&pwd={1}&persist=y' \
                                      .format(username, password)
        cookie = urlopen(target_url).read()
        return cookie

    def restore_ression(self, cookie):
        """Establish databse connection using permanent session cookie"""
        opener = build_opener()
        opener.addheaders.append(('Cookie', cookie))
        self._opener = opener

    def request_permanent_session(self, username=None, password=None):
        config = Config()
        if username is None and password is None:
            username, password = config.db_credentials
        cookie = self.request_sid_cookie(username, password)
        try:
            cookie_str = str(cookie, 'utf-8')  # Python 3
        except TypeError:
            cookie_str = str(cookie)  # Python 2
        log.debug("Session cookie: {0}".format(cookie_str))
        config.set('DB', 'session_cookie', cookie_str)
        self.restore_ression(cookie)

    def login(self, username, password):
        "Login to the databse and store cookies for upcoming requests."
        opener = self._build_opener()
        values = {'usr': username, 'pwd': password}
        req = self._make_request(self._login_url, values)
        try:
            f = opener.open(req)
        except URLError as e:
            log.error("Failed to connect to the database -> probably down!")
            log.error("Error from database server:\n    {0}".format(e))
            return False
        html = f.read()
        failed_auth_message = 'Bad username or password'
        if failed_auth_message in str(html):
            log.error(failed_auth_message)
            return False
        return True

    def _build_opener(self):
        cj = CookieJar()
        self._cookies = cj
        opener = build_opener(HTTPCookieProcessor(cj), HTTPHandler())
        return opener

    def _make_request(self, url, values):
        data = urlencode(values)
        return Request(url, data.encode('utf-8'))


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
        "Return a dict for given parameter"
        parameter = self._get_parameter_name(parameter)
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
        parameter = self._get_parameter_name(parameter).lower()
        return self._parameters[parameter]['Unit']

    def _get_parameter_name(self, name):
        if name in self.names:
            return name

        aliases = [n for n in self.names if n.endswith(' ' + name)]
        if len(aliases) == 1:
            log.info("Alias found for {0}: {1}".format(name, aliases[0]))
            return aliases[0]

        log.info("Parameter '{0}' not found, trying to find alternative."
                 .format(name))
        try:
            # ahrs_g[0] for example should be looked up as ahrs_g
            alternative = re.findall(r'(.*)\[[0-9*]\]', name)[0]
            log.info("Found alternative: '{0}'".format(alternative))
            return alternative
        except IndexError:
            raise KeyError("Could not find alternative for '{0}'"
                           .format(name))


class DOMContainer(object):
    """Provides easy access to DOM parameters stored in the DB."""
    def __init__(self, doms):
        self._json = doms
        self._ids = []

    def ids(self, det_id):
        """Return a list of DOM IDs for given detector"""
        return [dom['DOMId'] for dom in self._json if dom['DetOID'] == det_id]

    def clbupi2domid(self, clb_upi, det_id):
        """Return DOM ID for given CLB UPI and detector"""
        return self._json_list_lookup('CLBUPI', clb_upi, 'DOMId', det_id)

    def clbupi2floor(self, clb_upi, det_id):
        """Return Floor ID for given CLB UPI and detector"""
        return self._json_list_lookup('CLBUPI', clb_upi, 'Floor', det_id)

    def domid2floor(self, dom_id, det_id):
        """Return Floor ID for given DOM ID and detector"""
        return self._json_list_lookup('DOMId', dom_id, 'Floor', det_id)

    def via_omkey(self, omkey, det_id):
        """Return DOM for given OMkey (DU, floor)"""
        du, floor = omkey
        try:
            return DOM.from_json([d for d in self._json
                                  if d["DU"] == du and
                                  d["Floor"] == floor and
                                  d["DetOID"] == det_id][0])
        except IndexError:
            log.critical("No DOM found for OMKey '{0}' and DetOID '{1}'."
                         .format(omkey, det_id))

    def via_dom_id(self, dom_id):
        """Return DOM for given dom_id"""
        try:
            return DOM.from_json([d for d in self._json
                                  if d["DOMId"] == dom_id][0])
        except IndexError:
            log.critical("No DOM found for DOM ID '{0}'".format(dom_id))

    def via_clb_upi(self, clb_upi):
        """return DOM for given CLB UPI"""
        try:
            return DOM.from_json([d for d in self._json
                                  if d["CLBUPI"] == clb_upi][0])
        except IndexError:
            log.critical("No DOM found for CLB UPI '{0}'".format(clb_upi))

    def _json_list_lookup(self, from_key, value, target_key, det_id):
        lookup = [dom[target_key] for dom in self._json if
                  dom[from_key] == value and
                  dom['DetOID'] == det_id]
        if len(lookup) > 1:
            log.warn("Multiple entries found: {0}".format(lookup) + "\n" +
                     "Returning the first one.")
        return lookup[0]


class DOM(object):
    """Represents a DOM"""
    def __init__(self, clb_upi, dom_id, dom_upi, du, det_oid, floor):
        self.clb_upi = clb_upi
        self.dom_id = dom_id
        self.dom_upi = dom_upi
        self.du = du
        self.det_oid = det_oid
        self.floor = floor

    @classmethod
    def from_json(cls, json):
        return cls(json["CLBUPI"], json["DOMId"], json["DOMUPI"],
                   json["DU"], json["DetOID"], json["Floor"])

    def __str__(self):
        return "DU{0}-DOM{1}".format(self.du, self.floor)

    def __repr__(self):
        return ("{0} - DOM ID: {1}\n"
                "   DOM UPI: {2}\n"
                "   CLB UPI: {3}\n"
                "   DET OID: {4}\n"
                .format(self.__str__(), self.dom_id, self.dom_upi,
                        self.clb_upi, self.det_oid))
