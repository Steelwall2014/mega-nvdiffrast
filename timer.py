import logging
import time
import numpy as np

import torch
from distribute import log_dist

class CudaEventTimer(object):

    def __init__(self, start_event: torch.cuda.Event, end_event: torch.cuda.Event):
        self.start_event = start_event
        self.end_event = end_event

    def get_elapsed_msec(self):
        torch.cuda.current_stream().wait_event(self.end_event)
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)

def trim_mean(data, trim_percent):
    """Compute the trimmed mean of a list of numbers.

    Args:
        data (list): List of numbers.
        trim_percent (float): Percentage of data to trim.

    Returns:
        float: Trimmed mean.
    """
    assert 0.0 <= trim_percent <= 1.0
    n = len(data)
    # Account for edge case of empty list
    if len(data) == 0:
        return 0
    data.sort()
    k = int(round(n * (trim_percent)))
    return np.mean(data[k:n - k])
    
class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""

    class Timer:
        """Timer."""

        def __init__(self, name, use_host_timer=True):
            self.name_ = name
            self.started_ = False
            self.event_timers = []
            self.use_host_timer = use_host_timer
            self.start_event = None
            self.elapsed_records = None
            self.start_time = 0.0
            self.end_time = 0.0

        def start(self):
            """Start the timer."""
            if self.started_:
                self.reset()
            if self.use_host_timer:
                self.start_time = time.time()
            else:
                event_class = torch.cuda.Event
                self.start_event = event_class(enable_timing=True)
                self.start_event.record()
            self.started_ = True

        def stop(self, reset=False, record=False):
            """Stop the timer."""
            assert self.started_, "timer is not started"
            event_class = torch.cuda.Event
            if self.use_host_timer:
                self.end_time = time.time()
                self.event_timers.append(self.end_time - self.start_time)
            else:
                event_class = torch.cuda.Event
                end_event = event_class(enable_timing=True)
                end_event.record()
                self.event_timers.append(CudaEventTimer(self.start_event, end_event))
                self.start_event = None
            self.started_ = False

        def _get_elapsed_msec(self):
            if self.use_host_timer:
                self.elapsed_records = [et * 1000.0 for et in self.event_timers]
            else:
                self.elapsed_records = [et.get_elapsed_msec() for et in self.event_timers]
            return sum(self.elapsed_records)

        def reset(self):
            """Reset timer."""
            self.started_ = False
            self.start_event = None
            self.elapsed_records = None
            self.event_timers.clear()

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self._get_elapsed_msec()
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

        def mean(self):
            self.elapsed(reset=False)
            return trim_mean(self.elapsed_records, 0.1)
        
        def max(self):
            self.elapsed(reset=False)
            if len(self.elapsed_records) == 0:
                return 0
            return max(self.elapsed_records)

    def __init__(self):
        self.timers = {}
        self.log_enabled = True

    def get_timers(self):
        return self.timers

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    @staticmethod
    def memory_usage():
        alloc = "mem_allocated: {:.4f} GB".format(torch.cuda.memory_allocated() / (1024 * 1024 * 1024))
        max_alloc = "max_mem_allocated: {:.4f} GB".format(torch.cuda.max_memory_allocated() /
                                                          (1024 * 1024 * 1024))
        reserve = "reserve_allocated: {:.4f} GB".format(torch.cuda.memory_reserved() / (1024 * 1024 * 1024))
        max_reserve = "max_reserve_allocated: {:.4f} GB".format(torch.cuda.max_memory_reserved() /
                                                            (1024 * 1024 * 1024))
        return " | {} | {} | {} | {}".format(alloc, max_alloc, reserve, max_reserve)

    def log(self, names, normalizer=1.0, reset=True, memory_breakdown=False, ranks=None):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = f"time (ms)"
        for name in names:
            if name in self.timers:
                elapsed_time = (self.timers[name].elapsed(reset=reset) / normalizer)
                string += " | {}: {:.2f}".format(name, elapsed_time)

        log_dist(string, ranks=ranks or ["all"], level=logging.INFO)

    def get_mean(self, names, normalizer=1.0, reset=True):
        """Get the mean of a group of timers."""
        assert normalizer > 0.0
        means = {}
        for name in names:
            if name in self.timers:
                elapsed_time = (self.timers[name].mean() / normalizer)
                means[name] = elapsed_time
                if reset:
                    self.timers[name].reset()
        return means
    
    def get_max(self, names, normalizer=1.0, reset=True):
        """Get the max of a group of timers."""
        assert normalizer > 0.0
        maxs = {}
        for name in names:
            if name in self.timers:
                elapsed_time = (self.timers[name].max() / normalizer)
                maxs[name] = elapsed_time
                if reset:
                    self.timers[name].reset()
        return maxs

    def start(self, name, use_host_timer=True):
        if name not in self.timers:
            self.timers[name] = self.Timer(name, use_host_timer)
        self.timers[name].start()
    
    def stop(self, name):
        self.timers[name].stop()
        if self.log_enabled:
            self.log([name], reset=False)

    def enable_log(self, enable):
        self.log_enabled = enable
        
timers = SynchronizedWallClockTimer()