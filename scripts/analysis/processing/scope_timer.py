import time


class ScopeTimer:
    def __init__(self, scope_name="", print_time=True):
        self.scope_name = scope_name
        self.print_time = print_time
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        if self.print_time:
            elapsed_time = self.end_time - self.start_time
            print(f"{self.scope_name} execution time: {elapsed_time} seconds")

    @property
    def time_in_ms(self):
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer has not been started or stopped properly.")
        elapsed_time = self.end_time - self.start_time
        return elapsed_time * 1000
