import numpy as np

from time import time
from collections import deque


def time_hms(time_float: float) -> tuple:
	hours = int(time_float // 3600)
	minutes = int((time_float - hours * 3600) // 60)
	seconds = int(time_float % 60)

	return hours, minutes, seconds


class Timer:
    def __init__(self, total_iter: int, dsize: int = 10):
        self.start_time = time()
        self.previous_time = time()
        self.current_time = time()
        self.total_iter = total_iter
        self.dsize = dsize
        self.time_list = deque(maxlen=dsize)
        self.current_iter = 0

    def update(self):
        self.previous_time = self.current_time
        self.current_time = time()
        self.time_list.append(self.current_time - self.previous_time)
        self.current_iter += 1

    def mean_time(self) -> float:
        return np.mean(self.time_list)

    def interval(self) -> float:
        return self.time_list[-1]

    def elapsed(self) -> float:
        return self.current_time - self.start_time

    def remaining(self) -> float:
        return self.mean_time() * (self.total_iter - self.current_iter)

    def to_hms(time_float):
        hours = int(time_float // 3600)
        minutes = int((time_float - hours * 3600) // 60)
        seconds = int(time_float % 60)

        return hours, minutes, seconds

    def reset(self):
        self.start_time = time()
        self.previous_time = time()
        self.current_time = time()
        self.time_list = deque(maxlen=self.dsize)
        self.current_iter = 0
