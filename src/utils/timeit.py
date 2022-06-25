import time
from contextlib import ContextDecorator


def timed(method, *args, **kwargs):
    """
    Get execution time for a method.
    """
    ts = time.time()
    print("TS", ts)
    result = method(*args, **kwargs)
    te = time.time()
    print("TE", te)
    return (result, (te - ts) * 1000)


class TimeLogger(ContextDecorator):
    start = None

    def __init__(self, message):
        super(TimeLogger, self).__init__()
        self.message = message

    def __enter__(self):
        print(self.message, end="", flush=True)
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = round(time.time() - self.start, 2)
        print(f"Done, took {elapsed_time} seconds.", flush=True)
