import numpy as np

TRACK_DTYPE = [
    ('pos', np.float64, (3, )),
    ('dir', np.float64, (3, )),
    ('time', np.float64),
    ('energy', np.float64),
    ('quality', np.float64),
]


RECOLNS_DTYPE = TRACK_DTYPE[:]
RECOLNS_DTYPE.extend([
    ('beta', np.float64),
    ('n_fits', np.uint16),
    ('max_likelihood', np.float64),
    ('n_compatible_solutions', np.uint16),
    ('n_hits', np.uint32),
    ('error_matrix', np.float64, (15, )),
])

JGANDALF_DTYPE = TRACK_DTYPE[:]
JGANDALF_DTYPE.extend([

])

AASHOWERFIT_DTYPE = TRACK_DTYPE[:]
AASHOWERFIT_DTYPE.extend([

])

QSTRATEGY_DTYPE = TRACK_DTYPE[:]
QSTRATEGY_DTYPE.extend([

])

DUSJ_DTYPE = TRACK_DTYPE[:]
DUSJ_DTYPE.extend([

])

recname_to_dtype = {
    'RecoLNS': RECOLNS_DTYPE,
    'JGandalf': JGANDALF_DTYPE,
    'AaShowerFit': AASHOWERFIT_DTYPE,
    'QStrategy': QSTRATEGY_DTYPE,
    'Dusj': DUSJ_DTYPE,
}
