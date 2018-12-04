import numpy as np
import km3pipe as kp


class APump(kp.Pump):
    def configure(self):
        self.index = 0

    def process(self, blob):
        data = {'a': self.index * np.arange(5), 'b': np.arange(5)**self.index}
        data2 = {
            'c': self.index * np.arange(10, dtype='f4') + 0.1,
            'd': np.arange(10, dtype='f4')**self.index + 0.2
        }
        print(data2)
        blob['Tablelike'] = kp.Table(data, h5loc='/tablelike', name='2D Table')
        print(blob['Tablelike'])
        blob['Columnwise'] = kp.Table(
            data2,
            h5loc='/columnwise',
            split_h5=True,
            name='Column-wise Split'
        )
        self.index += 1
        return blob


pipe = kp.Pipeline()
pipe.attach(APump)
pipe.attach(kp.io.HDF5Sink, filename='km3hdf5_example.h5')
pipe.drain(13)
