"""
A tool to extract data from KM3NeT ROOT files (offline format)
to HDF5 fast (hence the f).

Usage:
    h5extractf [options] FILENAME
    h5extractf (-h | --help)

Options:
    -o OUTFILE                  Output file.
    --without-full-reco         Don't include all reco tracks, only the best.
    --without-calibration       Don't include calibration information for offline hits.
    -h --help                   Show this screen.

"""
import time
import numpy as np
import h5py
import awkward as ak
import km3io
import km3pipe as kp

FORMAT_VERSION = np.string_("5.1")


def h5extractf(
    root_file, outfile=None, without_full_reco=False, without_calibration=False
):
    if without_calibration:
        calibration_fields = []
    else:
        calibration_fields = [
            "pos_x",
            "pos_y",
            "pos_z",
            "dir_x",
            "dir_y",
            "dir_z",
            "tdc",
        ]
    fields = {
        "event_info": [
            ("id", "event_id"),  # id gets renamed to event_id
            "run_id",
            ("t_sec", "timestamp"),
            ("t_ns", "nanoseconds"),
            ("mc_t", "mc_time"),
            "trigger_mask",
            "trigger_counter",
            "overlays",
            "det_id",
            "frame_index",
            "mc_run_id",
        ],
        # weights get put into event_info as well
        "event_info_weights": [
            "weight_w1",
            "weight_w2",
            "weight_w3",
            "weight_w4",
        ],
        "hits": [
            "channel_id",
            "dom_id",
            ("t", "time"),
            "tot",
            ("trig", "triggered"),
            *calibration_fields,
        ],
        "mc_hits": [
            "a",
            "origin",
            "pmt_id",
            ("t", "time"),
        ],
        "tracks": [
            "pos_x",
            "pos_y",
            "pos_z",
            "dir_x",
            "dir_y",
            "dir_z",
            "E",
            "t",
            ("len", "length"),
            "rec_type",
            ("lik", "likelihood"),
            "id",
        ],
        "mc_tracks": [
            "pos_x",
            "pos_y",
            "pos_z",
            "dir_x",
            "dir_y",
            "dir_z",
            ("E", "energy"),
            ("t", "time"),
            ("len", "length"),
            "pdgid",
            "id",
        ],
    }

    if outfile is None:
        outfile = root_file + ".h5"
    start_time = time.time()
    with h5py.File(outfile, "w") as f:
        with km3io.OfflineReader(root_file) as r:
            if r.header is not None:
                print("Processing header")
                f.create_dataset(
                    "raw_header",
                    data=kp.io.hdf5.header2table(r.header),
                )
            print("Processing event_info")
            np_event_info = _branch_to_numpy(r, fields["event_info"])
            np_weights = _ak_to_numpy(r.w, fields["event_info_weights"])
            np_event_info[0].update(np_weights[0])
            _to_hdf(f, "event_info", np_event_info)

            # TODO remove group_info once km3pipe does not require it anymore
            group_info = np.core.records.fromarrays(
                [np.arange(len(np_event_info[1]))], names=["group_id"]
            )
            f.create_dataset("group_info", data=group_info)

            print("Processing tracks")
            reco = f.create_group("reco")
            for branch_data in _yield_tracks(
                r.tracks, fields["tracks"], without_full_reco=without_full_reco
            ):
                _to_hdf(reco, *branch_data)

            for field_name in ("hits", "mc_hits", "mc_tracks"):
                if r[field_name] is None:
                    continue
                print("Processing", field_name)
                np_branch = _branch_to_numpy(r[field_name], fields[field_name])
                if np_branch[1].sum() == 0:
                    # empty branch, e.g. mc_hits for data files
                    continue
                _to_hdf(f, field_name, np_branch)
        f.attrs.create("format_version", FORMAT_VERSION)
        f.attrs.create("km3pipe", kp.__version__)
        f.attrs.create("origin", root_file)
    print("Completed in {:.1f} s".format(time.time() - start_time))


def _branch_to_numpy(branch, fields):
    """
    Read 1D or 2D fields from a branch and convert them to numpy arrays.

    Parameters
    ----------
    branch : km3io.rootio.Branch
        Branch of an offline file.
    fields : list
        The field names. If an entry is a tuple instead of a str,
        first entry is the field name, secound entry is its h5 name.

    Returns
    -------
    tuple
        Numpy-fied awkward array, length 2.
        First entry is a dict with fields as keys, and 1D np.arrays as values.
        Secound entry is a 1D np.array, the number of items for each event.

    """
    data, n_items = {}, None
    # would be better to read out all fields at once
    for field in fields:
        if isinstance(field, (tuple, list)):
            field, column = field
        else:
            column = field
        d = branch.__getattr__(field)
        n_dims = d.ndim
        if n_dims == 1:
            if n_items is None:
                n_items = np.ones(len(d), dtype="int64")
        elif n_dims == 2:
            if n_items is None:
                n_items = ak.num(d).to_numpy()
            d = ak.flatten(d)
        else:
            raise ValueError("Can not process field", field)
        data[column] = d.to_numpy()
    return data, n_items


def _ak_to_numpy(ak_array, fields):
    """
    Convert the given awkward array to a numpy table.

    Parameters
    ----------
    ak_array : awkward.Array
        The awkward array, 2D or 3D.
    fields : List
        The column names of the last axis of the array.

    Returns
    -------
    np_branch : tuple
        Numpy-fied awkward array. See output of _branch_to_numpy.

    """
    n_dims = ak_array.ndim - 1
    if n_dims == 1:
        n_items = np.ones(len(ak_array), dtype="int64")
    elif n_dims == 2:
        n_items = ak.num(ak_array).to_numpy()
        ak_array = ak.flatten(ak_array)
    else:
        raise ValueError("Can not process array")

    filled = np.ma.filled(
        ak.pad_none(ak_array, target=len(fields), axis=-1).to_numpy(),
        fill_value=np.nan,
    )
    return {fields[i]: filled[:, i] for i in range(len(fields))}, n_items


def _yield_tracks(tracks, fields, without_full_reco=False):
    """
    Yield info from the tracks branch.

    Parameters
    ----------
    tracks : km3io.rootio.Branch
        The tracks branch.
    fields : list
        The fields to read for each track.
    without_full_reco : bool
        Don't include all reco tracks, only the best.

    Yields
    ------
    name : str
        Name of how the tracks where chosen.
    np_branch : tuple
        Numpy-fied awkward array. See output of _branch_to_numpy.

    """
    track_fmap = {
        "best_jmuon": (
            km3io.definitions.reconstruction.JMUONPREFIT,
            km3io.tools.best_jmuon,
        ),
        "best_jshower": (
            km3io.definitions.reconstruction.JSHOWERPREFIT,
            km3io.tools.best_jshower,
        ),
        "best_dusjshower": (
            km3io.definitions.reconstruction.DUSJSHOWERPREFIT,
            km3io.tools.best_dusjshower,
        ),
        "best_aashower": (
            km3io.definitions.reconstruction.AASHOWERFITPREFIT,
            km3io.tools.best_aashower,
        ),
    }
    if not without_full_reco:
        track_fmap["tracks"] = (None, None)

    n_columns = max(km3io.definitions.fitparameters.values()) + 1

    for name, (stage, func) in track_fmap.items():
        if stage is None or stage in tracks.rec_stages:
            if func is None:
                sel_tracks = tracks
            else:
                sel_tracks = func(tracks)
            np_branch = _branch_to_numpy(sel_tracks, fields)

            np_fitinf = _ak_to_numpy(sel_tracks.fitinf, range(n_columns))
            for fitparam, idx in km3io.definitions.fitparameters.items():
                np_branch[0][fitparam] = np_fitinf[0][idx].astype("float32")

            yield name, np_branch


def _to_hdf(f, name, np_branch, with_group_id=True):
    """
    Save a numpy-fied awkward array as a dataset to hdf5.

    Parameters
    ----------
    f : h5py.File
        The opened h5 file.
    name : str
        Name of the dataset to create.
    np_branch : tuple
        The data to save. See output of _branch_to_numpy.
    with_group_id : bool
        If False (default), save with indices as a sperate dataset.
        If true, save with group_id as an additional column.

    """
    # TODO set default of with_group_id to False once km3pipe supports
    #  indexed recarrays
    hdf_settings = {
        "compression": "gzip",
        "compression_opts": 5,
        "shuffle": True,
        "fletcher32": True,
        "chunks": True,
    }
    data, n_items = np_branch
    if with_group_id:
        data["group_id"] = np.repeat(np.arange(len(n_items)), n_items)
    else:
        index = np.concatenate([[0], np.cumsum(n_items)[:-1]])
        indices = np.core.records.fromarrays(
            [index, n_items], names=["index", "n_items"]
        )
        f.create_dataset(name + "_indices", data=indices, **hdf_settings)

    f.create_dataset(
        name,
        data=np.core.records.fromarrays(list(data.values()), names=list(data.keys())),
        **hdf_settings,
    )


def main():
    from docopt import docopt

    args = docopt(__doc__, version=kp.version)
    h5extractf(
        args["FILENAME"],
        args["-o"],
        without_full_reco=args["--without-full-reco"],
        without_calibration=args["--without-calibration"],
    )


if __name__ == "__main__":
    main()
