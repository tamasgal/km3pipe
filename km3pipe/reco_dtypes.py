import numpy as np

MINIMAL_TRACK_DTYPE = [
    ('event_id', np.uint32),
    ('pos_x', np.float64,),
    ('pos_y', np.float64,),
    ('pos_z', np.float64,),
    ('dir_x', np.float64,),
    ('dir_y', np.float64,),
    ('dir_z', np.float64,),
]


FULL_TRACK_DTYPE = MINIMAL_TRACK_DTYPE[:]
FULL_TRACK_DTYPE.extend([
    ('time', np.float64),
    ('energy', np.float64),
    ('quality', np.float64),
    ('did_converge', np.bool_),
])


RECOLNS_DTYPE = FULL_TRACK_DTYPE[:]
RECOLNS_DTYPE.extend([
    ('beta', np.float64),
    ('n_fits', np.float64),
    ('max_likelihood', np.float64),
    ('n_compatible_solutions', np.float64),
    ('n_hits', np.float64),
    #('error_matrix', np.float64, (15, )),
])

JGANDALF_DTYPE = FULL_TRACK_DTYPE[:]
JGANDALF_DTYPE.extend([

])

AASHOWERFIT_DTYPE = FULL_TRACK_DTYPE[:]
AASHOWERFIT_DTYPE.extend([

])

QSTRATEGY_DTYPE = FULL_TRACK_DTYPE[:]
QSTRATEGY_DTYPE.extend([

])

DUSJ_DTYPE = FULL_TRACK_DTYPE[:]
DUSJ_DTYPE.extend([

])

recname_to_dtype = {
    'RecoLNS': sorted(RECOLNS_DTYPE),
    'JGandalf': sorted(JGANDALF_DTYPE),
    'AaShowerFit': sorted(AASHOWERFIT_DTYPE),
    'QStrategy': sorted(QSTRATEGY_DTYPE),
    'Dusj': sorted(DUSJ_DTYPE),
}
