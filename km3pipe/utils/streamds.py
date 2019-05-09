"""
Access the KM3NeT StreamDS DataBase service.

Usage:
    streamds
    streamds list
    streamds upload [-q] CSV_FILE
    streamds info STREAM
    streamds get [-f FORMAT] STREAM [PARAMETERS...]
    streamds (-h | --help)
    streamds --version

Options:
    STREAM      Name of the stream.
    CSV_FILE    Whitespace separated data for the runsummary tables.
    PARAMETERS  List of parameters separated by space (e.g. detid=29).
    -f FORMAT   Usually 'txt' for ASCII or 'text' for UTF-8 [default: txt].
    -q          Dryrun! This will upload the parameters with a TEST_ prefix.
    -h --help   Show this screen.

"""

import getpass
import os
import json
import requests
import pandas as pd
import km3pipe as kp

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = kp.logger.get_logger("streamds")

RUNSUMMARY_URL = "https://km3netdbweb.in2p3.fr/jsonds/runsummarynumbers/i"
REQUIRED_COLUMNS = set(['run', 'det_id', 'source'])


def print_streams():
    """Print all available streams with their full description"""
    sds = kp.db.StreamDS()
    sds.print_streams()


def print_info(stream):
    """Print the information about a stream"""
    sds = kp.db.StreamDS()
    sds.help(stream)


def get_data(stream, parameters, fmt):
    """Retrieve data for given stream and parameters, or None if not found"""
    sds = kp.db.StreamDS()
    if stream not in sds.streams:
        log.error("Stream '{}' not found in the database.".format(stream))
        return
    params = {}
    if parameters:
        for parameter in parameters:
            if '=' not in parameter:
                log.error(
                    "Invalid parameter syntax '{}'\n"
                    "The correct syntax is 'parameter=value'".
                    format(parameter)
                )
                continue
            key, value = parameter.split('=')
            params[key] = value
    data = sds.get(stream, fmt, **params)
    if data is not None:
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):
            print(data)
    else:
        sds.help(stream)


def available_streams():
    """Show a short list of available streams."""
    sds = kp.db.StreamDS()
    print("Available streams: ")
    print(', '.join(sorted(sds.streams)))


def upload_runsummary(csv_filename, dryrun=False):
    """Reads the CSV file and uploads its contents to the runsummary table"""
    print("Checking '{}' for consistency.".format(csv_filename))
    if not os.path.exists(csv_filename):
        log.critical("{} -> file not found.".format(csv_filename))
        return
    try:
        df = pd.read_csv(csv_filename, delim_whitespace=True)
    except pd.errors.EmptyDataError as e:
        log.error(e)
        return

    cols = set(df.columns)

    if not REQUIRED_COLUMNS.issubset(cols):
        log.error(
            "Missing columns: {}.".format(
                ', '.join(str(c) for c in REQUIRED_COLUMNS - cols)
            )
        )
        return

    parameters = cols - REQUIRED_COLUMNS
    if len(parameters) < 1:
        log.error("No parameter columns found.")
        return

    if len(df) == 0:
        log.critical("Empty dataset.")
        return

    print(
        "Found data for parameters: {}.".format(
            ', '.join(str(c) for c in parameters)
        )
    )
    print("Converting CSV data into JSON")
    if dryrun:
        log.warn("Dryrun: adding 'TEST_' prefix to parameter names")
        prefix = "TEST_"
    else:
        prefix = ""
    db = kp.db.DBManager()    # noqa
    df['det_id'] = df['det_id'].apply(db.to_det_oid)
    data = convert_runsummary_to_json(df, prefix=prefix)
    print("We have {:.3f} MB to upload.".format(len(data) / 1024**2))

    print("Requesting database session.")
    if kp.db.we_are_in_lyon():
        session_cookie = "sid=_kmcprod_134.158_lyo7783844001343100343mcprod1223user"    # noqa
    else:
        session_cookie = kp.config.Config().get('DB', 'session_cookie')
        if session_cookie is None:
            raise SystemExit("Could not restore DB session.")
    log.debug("Using the session cookie: {}".format(session_cookie))
    cookie_key, sid = session_cookie.split('=')
    print("Uploading the data to the database.")
    r = requests.post(
        RUNSUMMARY_URL, cookies={cookie_key: sid}, files={'datafile': data}
    )
    if r.status_code == 200:
        log.debug("POST request status code: {}".format(r.status_code))
        print("Database response:")
        db_answer = json.loads(r.text)
        for key, value in db_answer.items():
            print("  -> {}: {}".format(key, value))
        if db_answer['Result'] == 'OK':
            print("Upload successful.")
        else:
            log.critical("Something went wrong.")
    else:
        log.error("POST request status code: {}".format(r.status_code))
        log.critical("Something went wrong...")
        return


def convert_runsummary_to_json(
        df, comment='Uploaded via km3pipe.StreamDS', prefix='TEST_'
):
    """Convert a Pandas DataFrame with runsummary to JSON for DB upload"""
    data_field = []
    comment += ", by {}".format(getpass.getuser())
    for det_id, det_data in df.groupby('det_id'):
        runs_field = []
        data_field.append({"DetectorId": det_id, "Runs": runs_field})

        for run, run_data in det_data.groupby('run'):
            parameters_field = []
            runs_field.append({
                "Run": int(run),
                "Parameters": parameters_field
            })

            parameter_dict = {}
            for row in run_data.iterrows():
                for parameter_name in run_data.columns:
                    if parameter_name in REQUIRED_COLUMNS:
                        continue

                    if parameter_name not in parameter_dict:
                        entry = {'Name': prefix + parameter_name, 'Data': []}
                        parameter_dict[parameter_name] = entry
                    data_value = getattr(row[1], parameter_name)
                    try:
                        data_value = float(data_value)
                    except ValueError as e:
                        log.critical("Data values has to be floats!")
                        raise ValueError(e)
                    value = {
                        'S': str(getattr(row[1], 'source')),
                        'D': data_value
                    }
                    parameter_dict[parameter_name]['Data'].append(value)
            for parameter_data in parameter_dict.values():
                parameters_field.append(parameter_data)
    data_to_upload = {"Comment": comment, "Data": data_field}
    file_data_to_upload = json.dumps(data_to_upload)
    return file_data_to_upload


def main():
    from docopt import docopt
    args = docopt(__doc__)

    if args['info']:
        print_info(args['STREAM'])
    elif args['list']:
        print_streams()
    elif args['upload']:
        upload_runsummary(args['CSV_FILE'], args['-q'])
    elif args['get']:
        get_data(args['STREAM'], args['PARAMETERS'], fmt=args['-f'])
    else:
        available_streams()
