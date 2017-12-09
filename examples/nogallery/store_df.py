import km3pipe as kp
import numpy as np
import pandas as pd

dt = np.dtype([('hi', int), ('there', float)])
dat = np.ones(5, dtype=dt)
df = pd.DataFrame.from_records(dat)


def foo(blob):
    blob["FooBar"] = df
    return blob


p = kp.Pipeline()
p.attach(foo)
p.attach(kp.io.HDF5Sink, filename="test.h5")
p.drain(10)
