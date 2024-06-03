import sys
import time

class Timer:
    """Simple timer class used to report runtime.
    """

    def __init__(self, filename=None):
        self._start_time = None
        self.filename = filename

    @property
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
        if self.is_running:
            raise TimerError("Timer is running. Use .stop() to stop it")

        if mytitle != "":
            self._print(f" --- {mytitle} --- ")
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
        if not self.is_running:
            raise TimerError("Timer is not running. Use .start() to start it")
        elapsed_time = self.get_elapsed()
        self._start_time = None
        if mytitle is not None:
            self._print(f"{mytitle}: {elapsed_time:.4f} seconds\n")
        return float(elapsed_time)

    def now(self):
        """Return the time elapsed since timer was started, without stopping the timer

        Returns:
            (float): elapsed time in seconds
        """
        return float(self.get_elapsed())

    def _print(self, *args, **kwargs):
        """Print the timing to :attr:`filename` if specified, or to screen.

        All arguments and keyword arguments are passed to ``print()``
        """

        if self.filename is None:
            print(*args, **kwargs)
        else:
            with open(self.filename,'a') as file:
                print(*args, file=file, **kwargs)

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""
