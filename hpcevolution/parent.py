import time

import torch
import pickle
from pathlib import Path
import logging

from evolution.hpc_map_elites import HPCMapElites
from evolution.initialization_utils import get_initial_model
from hpcevolution.work import Work
from hpcevolution.constants import SLEEP_TIME, AVAILABLE_EXTENSION
from hpcevolution.result import Result



class Parent:
    def __init__(self, num_iter, run_name, score_strategy, game, stop_after, save_model, gvgai_version, num_threads, log_level, max_age, is_mortality,
        is_crossover, crossover_possibility, mutate_possibility, mepgd_possibility, is_mepgd, cmame):
        self.run_name = run_name
        self.run_name_with_params = f'{run_name}-{game}-iter-{num_iter}-strat-{score_strategy}-stop-after-{stop_after}'
        logging.basicConfig(filename='../logs/parent-{}.log'.format(self.run_name_with_params), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info('Initializing parent')
        self.scoring_strategy = score_strategy
        self.score_strategy = score_strategy
        self.game = game
        self.stop_after = stop_after

        init_iter = 1

        self.evaluated_so_far = 0
        self.count_loops = 0
        #self.total_to_evaluate = num_iter
        self.total_to_evaluate = 80
        self.results_per_generation = 50
        self.work_per_generation = 0
        self.next_generation = True
        pop_size = 100
        policy_net, init_model = get_initial_model(gvgai_version, game)
        population = []
        
        #population.append(policy_net)
        
        #for i in range(1,pop_size):
            #temp,_ = get_initial_model(gvgai_version,game)
            #self.population.append(temp)

        # set up directories
        parent = Path(__file__).parent
        working_dir = parent / self.run_name
        self.AVAILABLE_AGENTS_DIR = working_dir / 'available_agents'
        self.WORK_DIR = working_dir / 'work_todo'
        self.RESULTS_DIR = working_dir / 'results'

        self.AVAILABLE_AGENTS_DIR.mkdir(exist_ok=True, parents=True)
        self.WORK_DIR.mkdir(exist_ok=True, parents=True)
        self.RESULTS_DIR.mkdir(exist_ok=True, parents=True)

        # ready to go
        self.map_elites = HPCMapElites(policy_net,
                  init_model,
                  init_iter,
                  num_iter,
                  is_crossover,
                  mutate_possibility,
                  crossover_possibility,
                  is_mortality,
                  max_age,
                  is_mepgd,
                  mepgd_possibility,
                  gvgai_version=gvgai_version,
                  is_cmame=cmame)

    # import pickle
    # result = Result(run_data.model, '0-0-0-0-0', 10)
    # pickle.dump(result, open('/Users/bradwindsor/ms_projects/qd-gen/gameQD/hpcevolution/results/1234.result', 'wb'))
    def run(self):
        results = []
        while self.evaluated_so_far < self.total_to_evaluate:
            if self.count_loops % 200 == 0:
                logging.info('INTERMEDIATE PERFORMANCES')
                for solution in self.map_elites.population:
                    logging.info(str(solution[2]))
                    logging.info(str(solution[1]))
            children = self.get_available_children()
            logging.info('{} available children, {} evals run'.format(len(children), self.evaluated_so_far))
            #100 solutions generated initially 50 selected and in the next generation approximately 50 new solutions generated
            #if self.evaluated_so_far >=1:
                #self.results.per_generation = 50
            for child_name in children:
                if self.next_generation == True:
                    
                    if len(results) < self.results_per_generation:
                        logging.info('Generating work for child')
                        run_data = self.generate_run_data()
                        self.write_work_for_child(run_data, child_name)
                        self.work_per_generation +=1
                    else:
                        self.next_generation = False

            logging.info('collecting results')
            found_results = self.collect_written_results()
            for result in found_results:
                print(result)
                results.append(result)
            logging.info('%d results found', len(results))
            
            #Switch to pure population based
            #New added to population ^ add agents in above routine
            if len(results) >= self.results_per_generation:
                #self.work_per_generation -= len(results)
                for result in results:
                    #Update pop
                    self.update_map_elites_results(result)
                    #file.unlink()
                self.evaluated_so_far += 1
                self.next_generation = True
                results = []
                self.update_population()
            logging.info('sleeping %d', SLEEP_TIME)
            time.sleep(SLEEP_TIME)
            self.count_loops += 1

        logging.info('Logging final results')
        #logging.info(str(self.map_elites.population))
        for solution in self.map_elites.population:
            print(solution[2])
            print(solution[1])
            logging.info("")


    def get_available_children(self):
        active = self.AVAILABLE_AGENTS_DIR.glob('*' + AVAILABLE_EXTENSION)
        active_agent_ids = [agent.stem for agent in active]
        return active_agent_ids

    def collect_written_results(self):
        results_files = self.RESULTS_DIR.glob('*.pkl')
        results = []
        for file in results_files:
            logging.info('loading result %s', file.stem)
            try:
                #results.append((pickle.load(file.open('rb')), file))
                results.append(pickle.load(file.open('rb')))
                file.unlink()
            except Exception as e:
                logging.info('failed to load result %s', str(e))
        logging.info('...done loading results...')
        return results

    def update_map_elites_results(self, result: Result):
        self.map_elites.update_result(result.network, result.feature, result.fitness)
    
    def update_population(self):
        fronts = self.map_elites.non_dominated_sort()
        crowding_dists = self.map_elites.crowding_distance(fronts)
        self.map_elites.select_new_population(fronts, crowding_dists)
    
    def generate_run_data(self):
        model = self.map_elites.next_model()
        return Work(model, self.score_strategy, self.game, self.stop_after, self.run_name_with_params)

    def write_work_for_child(self, work, child_name):
        path = self.WORK_DIR / str('task-id-{}-child-{}'.format(self.work_per_generation, child_name))
        path = path.with_suffix('.pkl')
        logging.info('Writing work %s', str(path))
        pickle.dump(work, path.open('wb'))
