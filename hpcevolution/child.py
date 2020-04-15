import time
class Child:
    def __init__(self):
        pass

    def do_task(self):
        while True:
            self.signal_unavailable()
            task = self.parse_received_task()
            result = self.run_task(task)
            self.write_result(result)
            self.signal_available()

            time.sleep(5)

    def parse_received_task(self):
        # LOAD NN FROM DISK
        pass

    def run_task(self, task):
        #RUN TRAIN_DQN
        pass

    def write_result(self, result):
        # MAP AND SCORE
        pass

    def signal_available(self):
        # WRITE FILE "CHILD 1 AVAILABLE"
        pass

    def signal_unavailable(self):
        # DELETE FILE "CHILD 1 AVAILABLE"
        pass
