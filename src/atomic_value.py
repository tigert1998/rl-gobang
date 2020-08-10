import threading


class AtomicValue:
    def __init__(self, value=0):
        self.value = value
        self.lock = threading.Lock()

    def __add__(self, value):
        with self.lock:
            self.value += value

    def load(self):
        with self.lock:
            return self.value

    def store(self, value):
        with self.lock:
            self.value = value
