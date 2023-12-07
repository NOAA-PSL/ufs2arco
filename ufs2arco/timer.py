import sys
import time

class Timer:
    """Simple timer class used to report runtime.
    """

    def __init__(self, filename=None):
        self._start_time = None
        self.filename = filename

    def is_running(self):
        """If this timer is running, return True

        Returns:
            is_running (bool): if this timer has been started, return True, otherwise False
        """
        return self._start_time is not None

    def start(self, mytitle=""):
        """Start a new timer and print what is about to be timed

        Args:
            mytitle (str, optional): indicate the thing that's about to run and be timed
        """
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self.print(f" --- {mytitle} --- ")
        self._start_time = time.perf_counter()

    def get_elapsed(self):
        """Get time that has passed since timer started

        Returns:
            (float): amount of time elapsed in seconds
        """
        return time.perf_counter() - self._start_time

    def stop(self, mytitle="Elapsed time"):
        """Stop the timer, and report the elapsed time. Raise TimerError if
        no timer is running.

        Returns:
            (float): amount of time elapsed in seconds
        """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")
        elapsed_time = self.get_elapsed()
        self._start_time = None
        self.print(mytitle+f": {elapsed_time:0.4f} seconds\n")
        return float(elapsed_time)

    def print(self, mystr):
        """Print the timing to :attr:`filename` if specified, or to screen.

        Args:
            mystr (str): to be printed
        """

        if self.filename is None:
            print(mystr)
        else:
            with open(self.filename,'a') as file:
                print(mystr,file=file)

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""
