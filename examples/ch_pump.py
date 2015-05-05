import time

from km3pipe import Pipeline, Module
from km3pipe.pumps import CHPump

class Sleep(Module):
    def process(self, blob):
        time.sleep(5)
        return blob

class PrintBlob(Module):
    def process(self, blob):
        print blob
        return blob

pipe = Pipeline()
pipe.attach(CHPump, host='127.0.0.1')
pipe.attach(PrintBlob)
pipe.attach(Sleep)
pipe.drain(10)

