import time
import result
import pickle
import os
from models.train_dqn import evaluate_net
from hpcevolution.constants import SLEEP_TIME, ACTIVE_AGENTS_DIR_PATHLIB, ACTIVE_EXTENSION, RESULTS_DIR_PATHLIB, \
    WORK_DIR_PATHLIB


def combine_scores(scores, score, win, mode):
    if mode == SCORE_ALL:
        scores += score
    elif mode == SCORE_WINNING:
        if win == 1:
            scores += score
    elif mode == SCORE_LOSING:
        if win == 0:
            scores += score
    return scores

def fitness_feature_fn(score_strategy, stop_after, game, run_name, policy_net, env_maker):
    """
    Calculate fitess and feature descriptor simultaneously
    """
    scores = 0
    wins = []
    for lvl in range(5):
        score, win = evaluate_net(policy_net,
                                  game_level=f'{game}-lvl{lvl}-v0',
                                  stop_after=stop_after,
                                  env_maker=env_maker)
        scores = combine_scores(scores, score, win, score_strategy)
        wins.append(win)

    fitness = scores
    feature_descriptor = '-'.join([str(i) for i in wins])
    return fitness, feature_descriptor

class Child:
    def __init__(self, unique_id, env_maker):
        self.id = unique_id
        self.is_available = False
        self.env_maker = env_maker
        pass

    def do_task(self):
        while True:
            self.signal_unavailable()
            task = self.parse_received_task()
            fitness, feature = self.run_task(task)
            res = Result(task.run_name,task.model,feature,fitness)
            self.write_result(res)
            self.signal_available()

            time.sleep(5)

    def parse_received_task(self):
        # LOAD NN FROM DISK
        path = WORK_DIR_PATHLIB / child_name  + '.pkl'
        task = pickle.load(open(path, 'rb'))
        return task

    def run_task(self, task):
        #RUN TRAIN_DQN

        fitness, feature = fitness_feature_fn(task.score_strategy, task.stop_after, task.game, 
                                                        run_name, task.model, self.env_maker)
        return feature,fitness
        

    def write_result(self, result):
        # MAP AND SCORE
        path = RESULTS_DIR_PATHLIB / self.id  + '.pkl'
        pickle.dump(result, path.open())
        return

    def signal_available(self):
        # WRITE FILE "CHILD 1 AVAILABLE"
        fn = "child_" + self.id + ".active"
        f = open(fn,"x")
        return

    def signal_unavailable(self):
        # DELETE FILE "CHILD 1 AVAILABLE"
        fn = "child_" + self.id + ".active"

        try:
            os.remove(fn)
        except OSError:
            pass

        return
