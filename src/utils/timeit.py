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

    def __init__(self, enter_msg: str, exit_msg="Done, took %f seconds."):
        """
        Args:
            enter_msg: string to be printed when starts the execution
            exit_msg: string to be printed when ends the execution. Must contain
                a `%f` formatter where set the elapsed time.
        """
        super(TimeLogger, self).__init__()
        self.enter_msg = enter_msg
        self.exit_msg = exit_msg

    def __enter__(self):
        print(self.enter_msg, end="", flush=True)
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = round(time.time() - self.start, 2)
        print(self.exit_msg % elapsed_time, flush=True)
