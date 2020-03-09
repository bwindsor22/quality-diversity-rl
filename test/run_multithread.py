import time
import threading
import random
import logging
from models.lockable_resource import LockableResource
class SimpleResource:
    def __call__(self, args):
        print(args)

def worker(resource):
    """thread worker function"""
    resource.acquire()
    pause = random.randint(1, 30)
    print('sleeping %s' % pause)
    time.sleep(pause)
    print('calling resource')
    resource(pause)
    print('ending')
    resource.release()
    return

lockables = [LockableResource(SimpleResource()) for _ in range(3)]

main_thread = threading.currentThread()
for _ in range(100):
    time.sleep(3)
    active_threads = [t for t in threading.enumerate() if not t is main_thread]
    print(len(active_threads), 'active')
    if len(active_threads) <= 7:
        print('trying to start new')
        unlocked = [l for l in lockables if not l.is_locked()]
        if len(unlocked):
            print(len(unlocked), 'unlocked resources')
            resource = unlocked[0]
            print('starting new thread with acquired resource')
            t = threading.Thread(target=worker, args=[resource])
            t.start()
        else:
            print('all resources locked, waiting')

