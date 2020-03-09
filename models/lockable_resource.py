import threading

class LockableResource:
    def __init__(self, resource):
        self.lock = threading.Lock()
        self.resource = resource

    def __call__(self, *args):
        return self.resource(*args)

    def acquire(self):
        self.lock.acquire()


    def release(self):
        self.lock.release()

    def is_locked(self):
        return self.lock.locked()
