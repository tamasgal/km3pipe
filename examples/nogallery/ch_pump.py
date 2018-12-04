from km3pipe import Pipeline, Module
from km3pipe.io import CHPump


class CHPrinter(Module):
    def process(self, blob):
        print("New blob:")
        print(blob['CHPrefix'])
        print(blob['CHData'])
        return blob


pipe = Pipeline()
pipe.attach(
    CHPump,
    host='127.0.0.1',
    port=5553,
    tags="foo, narf",
    timeout=1000,
    max_queue=42
)
pipe.attach(CHPrinter)
pipe.drain()
