# Filename: dataclasses.py
# pylint: disable=W0232,C0103,C0111
# vim:set ts=4 sts=4 sw=4 et syntax=python:
"""
Dataclasses for internal use. Heavily based on Numpy arrays.
"""
import itertools

import numpy as np
from numpy.lib import recfunctions as rfn

from .dataclass_templates import TEMPLATES
from .logger import get_logger
from .tools import istype

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
__all__ = ("Table", "is_structured", "has_structured_dt", "inflate_dtype")

DEFAULT_H5LOC = "/misc"
DEFAULT_NAME = "Generic Table"
DEFAULT_SPLIT = False
DEFAULT_H5SINGLETON = False

log = get_logger(__name__)


def has_structured_dt(arr):
    """Check if the array representation has a structured dtype."""
    arr = np.asanyarray(arr)
    return is_structured(arr.dtype)


def is_structured(dt):
    """Check if the dtype is structured."""
    if not hasattr(dt, "fields"):
        return False
    return dt.fields is not None


def inflate_dtype(arr, names):
    """Create structured dtype from a 2d ndarray with unstructured dtype."""
    arr = np.asanyarray(arr)
    if has_structured_dt(arr):
        return arr.dtype
    s_dt = arr.dtype
    dt = [(n, s_dt) for n in names]
    dt = np.dtype(dt)
    return dt


class Table(np.recarray):
    """2D generic Table with grouping index.

    This is a `np.recarray` subclass with some metadata and helper methods.

    You can initialize it directly from a structured numpy array,
    a pandas DataFrame, a dictionary of (columnar) arrays; or, initialize it
    from a list of rows/list of columns using the appropriate factory.

    This class adds the following to ``np.recarray``:

    Parameters
    ----------
    data: array-like or dict(array-like)
        numpy array with structured/flat dtype, or dict of arrays.
    h5loc: str
        Location in HDF5 file where to store the data. [default: '/misc']
    h5singleton: bool
        Tables defined as h5singletons are only written once to an HDF5 file.
        This is used for headers for example (default=False).
    dtype: numpy dtype
        Datatype over array. If not specified and data is an unstructured
        array, ``names`` needs to be specified. [default: None]

    Attributes
    ----------
    h5loc: str
        HDF5 group where to write into. (default='/misc')
    split_h5: bool
        Split the array into separate arrays, column-wise, when saving
        to hdf5? (default=False)
    name: str
        Human-readable name, e.g. 'Hits'
    h5singleton: bool
        Tables defined as h5singletons are only written once to an HDF5 file.
        This is used for headers for example (default=False).

    Methods
    -------
    from_dict(arr_dict, dtype=None, **kwargs)
        Create an Table from a dict of arrays (similar to pandas).
    from_template(data, template, **kwargs)
        Create an array from a dict of arrays with a predefined dtype.
    sorted(by)
        Sort the table by one of its columns.
    append_columns(colnames, values)
        Append new columns to the table.
    to_dataframe()
        Return as pandas dataframe.
    from_dataframe(df, **kwargs)
        Instantiate from a dataframe.
    from_rows(list_of_rows, **kwargs)
        Instantiate from an array-like with shape (n_rows, n_columns).
    from_columns(list_of_columns, **kwargs)
        Instantiate from an array-like with shape (n_columns, n_rows).
    """

    def __new__(
        cls,
        data,
        h5loc=DEFAULT_H5LOC,
        dtype=None,
        split_h5=DEFAULT_SPLIT,
        name=DEFAULT_NAME,
        h5singleton=DEFAULT_H5SINGLETON,
        **kwargs
    ):
        if isinstance(data, dict):
            return cls.from_dict(
                data,
                h5loc=h5loc,
                dtype=dtype,
                split_h5=split_h5,
                name=name,
                h5singleton=h5singleton,
                **kwargs
            )
        if istype(data, "DataFrame"):
            return cls.from_dataframe(
                data,
                h5loc=h5loc,
                dtype=dtype,
                split_h5=split_h5,
                name=name,
                h5singleton=h5singleton,
                **kwargs
            )
        if isinstance(data, (list, tuple)):
            raise ValueError(
                "Lists/tuples are not supported! "
                "Please use the `from_rows` or `from_columns` method instead!"
            )
        if isinstance(data, np.record):
            # single record from recarrary/kp.Tables, let's blow it up
            data = data[np.newaxis]
        if not has_structured_dt(data):
            # flat (nonstructured) dtypes fail miserably!
            # default to `|V8` whyever
            raise ValueError(
                "Arrays without structured dtype are not supported! "
                "Please use the `from_rows` or `from_columns` method instead!"
            )

        if dtype is None:
            dtype = data.dtype

        assert is_structured(dtype)

        if dtype != data.dtype:
            dtype_names = set(dtype.names)
            data_dtype_names = set(data.dtype.names)
            if dtype_names == data_dtype_names:
                if not all(dtype[f] == data.dtype[f] for f in dtype_names):
                    log.critical(
                        "dtype mismatch! Matching field names but differing "
                        "field types, no chance to reorder.\n"
                        "dtype of data:   %s\n"
                        "requested dtype: %s" % (data.dtype, dtype)
                    )
                    raise ValueError("dtype mismatch")
                log.once(
                    "dtype mismatch, but matching field names and types. "
                    "Rordering input data...",
                    identifier=h5loc,
                )
                data = Table({f: data[f] for f in dtype_names}, dtype=dtype)
            else:
                log.critical(
                    "dtype mismatch, no chance to reorder due to differing "
                    "fields!\n"
                    "dtype of data:   %s\n"
                    "requested dtype: %s" % (data.dtype, dtype)
                )
                raise ValueError("dtype mismatch")

        obj = np.asanyarray(data, dtype=dtype).view(cls)
        obj.h5loc = h5loc
        obj.split_h5 = split_h5
        obj.name = name
        obj.h5singleton = h5singleton
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            # called from explicit contructor
            return obj
        # views or slices
        self.h5loc = getattr(obj, "h5loc", DEFAULT_H5LOC)
        self.split_h5 = getattr(obj, "split_h5", DEFAULT_SPLIT)
        self.name = getattr(obj, "name", DEFAULT_NAME)
        self.h5singleton = getattr(obj, "h5singleton", DEFAULT_H5SINGLETON)
        # attribute access returns void instances on slicing/iteration
        # kudos to
        # https://github.com/numpy/numpy/issues/3581#issuecomment-108957200
        if obj is not None and type(obj) is not type(self):
            self.dtype = np.dtype((np.record, obj.dtype))

    def __array_wrap__(self, out_arr, context=None):
        # then just call the parent
        return Table(
            np.recarray.__array_wrap__(self, out_arr, context),
            h5loc=self.h5loc,
            split_h5=self.split_h5,
            name=self.name,
            h5singleton=self.h5singleton,
        )

    @staticmethod
    def _expand_scalars(arr_dict):
        scalars = []
        maxlen = 1  # have at least 1-elem arrays
        for k, v in arr_dict.items():
            if np.isscalar(v):
                scalars.append(k)
                continue
            # TODO: this is not covered yet, don't know if we need this
            # if hasattr(v, 'shape') and v.shape == (1,):  # np.array([1])
            #     import pdb; pdb.set_trace()
            #     arr_dict[k] = v[0]
            #     continue
            if hasattr(v, "ndim") and v.ndim == 0:  # np.array(1)
                arr_dict[k] = v.item()
                continue
            if len(v) > maxlen:
                maxlen = len(v)
        for s in scalars:
            arr_dict[s] = np.full(maxlen, arr_dict[s])
        return arr_dict

    @classmethod
    def from_dict(cls, arr_dict, dtype=None, fillna=False, **kwargs):
        """Generate a table from a dictionary of arrays."""
        arr_dict = arr_dict.copy()
        # i hope order of keys == order or values
        if dtype is None:
            names = sorted(list(arr_dict.keys()))
        else:
            dtype = np.dtype(dtype)
            dt_names = [f for f in dtype.names]
            dict_names = [k for k in arr_dict.keys()]
            missing_names = set(dt_names) - set(dict_names)
            if missing_names:
                if fillna:
                    dict_names = dt_names
                    for missing_name in missing_names:
                        arr_dict[missing_name] = np.nan
                else:
                    raise KeyError("Dictionary keys and dtype fields do not match!")
            names = list(dtype.names)

        arr_dict = cls._expand_scalars(arr_dict)
        data = [arr_dict[key] for key in names]
        return cls(np.rec.fromarrays(data, names=names, dtype=dtype), **kwargs)

    @classmethod
    def from_columns(cls, column_list, dtype=None, colnames=None, **kwargs):
        if dtype is None or not is_structured(dtype):
            # infer structured dtype from array data + column names
            if colnames is None:
                raise ValueError(
                    "Need to either specify column names or a "
                    "structured dtype when passing unstructured arrays!"
                )
            dtype = inflate_dtype(column_list, colnames)
            colnames = dtype.names
        if len(column_list) != len(dtype.names):
            raise ValueError("Number of columns mismatch between data and dtype!")
        data = {k: column_list[i] for i, k in enumerate(dtype.names)}
        return cls(data, dtype=dtype, colnames=colnames, **kwargs)

    @classmethod
    def from_rows(cls, row_list, dtype=None, colnames=None, **kwargs):
        if dtype is None or not is_structured(dtype):
            # infer structured dtype from array data + column names
            if colnames is None:
                raise ValueError(
                    "Need to either specify column names or a "
                    "structured dtype when passing unstructured arrays!"
                )
            dtype = inflate_dtype(row_list, colnames)
        # this *should* have been checked above, but do this
        # just to be sure in case I screwed up the logic above;
        # users will never see this, this should only show in tests
        assert is_structured(dtype)
        data = np.asanyarray(row_list).view(dtype)
        # drop useless 2nd dim
        data = data.reshape((data.shape[0],))
        return cls(data, **kwargs)

    @property
    def templates_avail(self):
        return sorted(list(TEMPLATES.keys()))

    @classmethod
    def from_template(cls, data, template):
        """Create a table from a predefined datatype.

        See the ``templates_avail`` property for available names.

        Parameters
        ----------
        data
            Data in a format that the ``__init__`` understands.
        template: str or dict
            Name of the dtype template to use from ``kp.dataclasses_templates``
            or a ``dict`` containing the required attributes (see the other
            templates for reference).
        """
        name = DEFAULT_NAME
        if isinstance(template, str):
            name = template
            table_info = TEMPLATES[name]
        else:
            table_info = template
        if "name" in table_info:
            name = table_info["name"]
        dt = table_info["dtype"]
        loc = table_info["h5loc"]
        split = table_info["split_h5"]
        h5singleton = table_info["h5singleton"]

        return cls(
            data,
            h5loc=loc,
            dtype=dt,
            split_h5=split,
            name=name,
            h5singleton=h5singleton,
        )

    @staticmethod
    def _check_column_length(values, n):
        values = np.atleast_2d(values)
        for v in values:
            if len(v) == n:
                continue
            else:
                raise ValueError(
                    "Trying to append more than one column, but "
                    "some arrays mismatch in length!"
                )

    def append_columns(self, colnames, values, **kwargs):
        """Append new columns to the table.

        When appending a single column, ``values`` can be a scalar or an
        array of either length 1 or the same length as this array (the one
        it's appended to). In case of multiple columns, values must have
        the shape ``list(arrays)``, and the dimension of each array
        has to match the length of this array.

        See the docs for ``numpy.lib.recfunctions.append_fields`` for an
        explanation of the remaining options.
        """
        n = len(self)
        if np.isscalar(values):
            values = np.full(n, values)

        values = np.atleast_1d(values)
        if not isinstance(colnames, str) and len(colnames) > 1:
            values = np.atleast_2d(values)
            self._check_column_length(values, n)

        if values.ndim == 1:
            if len(values) > n:
                raise ValueError("New Column is longer than existing table!")
            elif len(values) > 1 and len(values) < n:
                raise ValueError(
                    "New Column is shorter than existing table, "
                    "but not just one element!"
                )
            elif len(values) == 1:
                values = np.full(n, values[0])
        new_arr = rfn.append_fields(
            self, colnames, values, usemask=False, asrecarray=True, **kwargs
        )
        return self.__class__(
            new_arr,
            h5loc=self.h5loc,
            split_h5=self.split_h5,
            name=self.name,
            h5singleton=self.h5singleton,
        )

    def drop_columns(self, colnames, **kwargs):
        """Drop  columns from the table.

        See the docs for ``numpy.lib.recfunctions.drop_fields`` for an
        explanation of the remaining options.
        """
        new_arr = rfn.drop_fields(
            self, colnames, usemask=False, asrecarray=True, **kwargs
        )
        return self.__class__(
            new_arr,
            h5loc=self.h5loc,
            split_h5=self.split_h5,
            name=self.name,
            h5singleton=self.h5singleton,
        )

    def sorted(self, by, **kwargs):
        """Sort array by a column.

        Parameters
        ==========
        by: str
            Name of the columns to sort by(e.g. 'time').
        """
        sort_idc = np.argsort(self[by], **kwargs)
        return self.__class__(
            self[sort_idc], h5loc=self.h5loc, split_h5=self.split_h5, name=self.name
        )

    def to_dataframe(self):
        from pandas import DataFrame

        return DataFrame(self)

    @classmethod
    def from_dataframe(cls, df, **kwargs):
        rec = df.to_records(index=False)
        return cls(rec, **kwargs)

    @classmethod
    def merge(cls, tables, fillna=False):
        """Merge a list of tables"""
        cols = set(itertools.chain(*[table.dtype.descr for table in tables]))

        tables_to_merge = []
        for table in tables:
            missing_cols = cols - set(table.dtype.descr)

            if missing_cols:
                if fillna:
                    n = len(table)
                    n_cols = len(missing_cols)
                    col_names = []
                    for col_name, col_dtype in missing_cols:
                        if "f" not in col_dtype:
                            raise ValueError(
                                "Cannot create NaNs for non-float"
                                " type column '{}'".format(col_name)
                            )
                        col_names.append(col_name)

                    table = table.append_columns(
                        col_names, np.full((n_cols, n), np.nan)
                    )
                else:
                    raise ValueError(
                        "Table columns do not match. Use fill_na=True"
                        " if you want to append missing values with NaNs"
                    )
            tables_to_merge.append(table)

        first_table = tables_to_merge[0]

        merged_table = sum(tables_to_merge[1:], first_table)

        merged_table.h5loc = first_table.h5loc
        merged_table.h5singleton = first_table.h5singleton
        merged_table.split_h5 = first_table.split_h5
        merged_table.name = first_table.name

        return merged_table

    def __add__(self, other):
        cols1 = set(self.dtype.descr)
        cols2 = set(other.dtype.descr)
        if len(cols1 ^ cols2) != 0:
            cols1 = set(self.dtype.names)
            cols2 = set(other.dtype.names)
            if len(cols1 ^ cols2) == 0:
                raise NotImplementedError
            else:
                raise TypeError("Table columns do not match")
        col_order = list(self.dtype.names)
        ret = self.copy()
        len_self = len(self)
        len_other = len(other)
        final_length = len_self + len_other
        ret.resize(final_length, refcheck=False)
        ret[len_self:] = other[col_order]
        return Table(
            ret,
            h5loc=self.h5loc,
            h5singleton=self.h5singleton,
            split_h5=self.split_h5,
            name=self.name,
        )

    def __str__(self):
        name = self.name
        spl = "split" if self.split_h5 else "no split"
        s = "{} {}\n".format(name, type(self))
        s += "HDF5 location: {} ({})\n".format(self.h5loc, spl)
        s += "\n".join(
            map(
                lambda d: "{1} (dtype: {2}) = {0}".format(self[d[0]], *d),
                self.dtype.descr,
            )
        )
        return s

    def __repr__(self):
        s = "{} {} (rows: {})".format(self.name, type(self), self.size)
        return s

    def __contains__(self, elem):
        return elem in self.dtype.names

    @property
    def pos(self):
        return np.array([self.pos_x, self.pos_y, self.pos_z]).T

    @pos.setter
    def pos(self, arr):
        arr = np.atleast_2d(arr)
        assert arr.shape[1] == 3
        assert len(arr) == len(self)
        self.pos_x = arr[:, 0]
        self.pos_y = arr[:, 1]
        self.pos_z = arr[:, 2]

    @property
    def dir(self):
        return np.array([self.dir_x, self.dir_y, self.dir_z]).T

    @dir.setter
    def dir(self, arr):
        arr = np.atleast_2d(arr)
        assert arr.shape[1] == 3
        assert len(arr) == len(self)
        self.dir_x = arr[:, 0]
        self.dir_y = arr[:, 1]
        self.dir_z = arr[:, 2]

    @property
    def phi(self):
        from km3pipe.math import phi_separg

        return phi_separg(self.dir_x, self.dir_y)

    @property
    def theta(self):
        from km3pipe.math import theta_separg

        return theta_separg(self.dir_z)

    @property
    def zenith(self):
        from km3pipe.math import neutrino_to_source_direction

        _, zen = neutrino_to_source_direction(self.phi, self.theta)
        return zen

    @property
    def azimuth(self):
        from km3pipe.math import neutrino_to_source_direction

        azi, _ = neutrino_to_source_direction(self.phi, self.theta)
        return azi

    @property
    def triggered_rows(self):
        if not hasattr(self, "triggered"):
            raise KeyError("Table has no 'triggered' column!")
        return self[self.triggered.astype(bool)]


class NDArray(np.ndarray):
    """Array with HDF5 metadata."""

    def __new__(cls, array, dtype=None, order=None, **kwargs):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)
        h5loc = kwargs.get("h5loc", "/misc")
        title = kwargs.get("title", "Unnamed NDArray")
        group_id = kwargs.get("group_id", None)
        obj.h5loc = h5loc
        obj.title = title
        obj.group_id = group_id
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.h5loc = getattr(obj, "h5loc", None)
        self.title = getattr(obj, "title", None)
        self.group_id = getattr(obj, "group_id", None)


class Vec3(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(*np.add(self, other))

    def __radd__(self, other):
        return Vec3(*np.add(other, self))

    def __sub__(self, other):
        return Vec3(*np.subtract(self, other))

    def __rsub__(self, other):
        return Vec3(*np.subtract(other, self))

    def __mul__(self, other):
        return Vec3(*np.multiply(self, other))

    def __rmul__(self, other):
        return Vec3(*np.multiply(other, self))

    def __div__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        return Vec3(*np.divide(self, other))

    def __array__(self, dtype=None):
        if dtype is not None:
            return np.array([self.x, self.y, self.z], dtype=dtype)
        else:
            return np.array([self.x, self.y, self.z])

    def __getitem__(self, index):
        return self.__array__()[index]
