import time


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
