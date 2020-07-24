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
        self.total_to_evaluate = 1500
        self.results_per_generation = 400
        self.work_per_generation = 0
        self.next_generation = True
        pop_size = 400
        policy_net, init_model = get_initial_model(gvgai_version, game)
        population = []
        self.num_levels = 2
        self.objectives = [[5],[7]]
        
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
        self.fronts = []
        self.crowding_dists = []
        self.levels = [5,7,8,4,9,3]

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
                i = 0
                logging.info('INTERMEDIATE PERFORMANCES')
                for solution in self.map_elites.population:
                    logging.info(str(solution[2]))
                    logging.info(str(solution[1]))

                logging.info("fronts")

                for front in self.fronts:
                    logging.info("front: " + str(i))
                    for p in front:
                        logging.info(p[2])
                        logging.info(p[1])
                    i += 1

                logging.info("crowding distances")

                for crowding_dist in self.crowding_dists:
                    for itemid, distance in crowding_dist.items():
                        logging.info(itemid)
                        logging.info(distance)

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
            
            if self.evaluated_so_far % 250 == 0 and self.evaluated_so_far != 0:
                logging.info("Intermediate Results : SAVING MODELS")
                j = 0
                for solution in self.map_elites.population:
                    logging.info(solution[2])
                    logging.info(solution[1])
                    logging.info("")
                    torch.save(solution[0],'/scratch/bos221/qd-branches/nsgaii-multi-level/quality-diversity-rl/saved_models/torch_model_{}_{}'.format(solution[2],str(j)))
                    j +=1

        logging.info('Logging final results')
        #logging.info(str(self.map_elites.population))
        for solution in self.map_elites.population:
            logging.info(solution[2])
            logging.info(solution[1])
            logging.info("")
            torch.save(solution[0],'/scratch/bos221/qd-branches/nsgaii-multi-level/quality-diversity-rl/saved_models/torch_model_{}'.format(solution[2]))
        i = 0
        for front in self.fronts:
            logging.info("front: " + str(i)) 
            for p in front:
                logging.info(p[2])
                logging.info(p[1])
            i += 1

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
                temp = pickle.load(file.open('rb'))
                if temp.num_levels == self.num_levels:
                    results.append(temp)
                file.unlink()
            except Exception as e:
                logging.info('failed to load result %s', str(e))
        logging.info('...done loading results...')
        return results

    def update_map_elites_results(self, result: Result):
        self.map_elites.update_result(result.network, result.feature, result.fitness,result.num_levels)
  

    def reevaluate_population(self):
       for solution in self.map_elites.population:
           #write work with new objectives using old model
           new_work = Work(model, self.score_strategy, self.game, self.stop_after, self.run_name_with_params, self.num_levels, self.objectives)
           #find available children
           #write work for available child           

    def update_objectives(self):
       self.num_levels +=1
       self.objectives[0].append(objectives[1][0])
       self.objectives[1][0] = self.levels[self.num_levels - 1]
       self.reevaluate_population()


    def check_objectives(self):
        #Alternatively threshold of around 10 should work as well
        won_all_levels = ""

        for i in range(0,self.num_levels):
            won_all_levels += "1-1"

        for solution in self.map_elites.population:
            if solution[2] == won_all_levels:
                found_winning_agent = True
                break
        
        if found_winning_agent == True:
            self.update_objectives()



    def update_population(self):

        #Remove non-updated models from population
        filtered_population = []
        for solution in self.population:
            if solution[3] == self.num_levels:
                filtered_population.append(solution)

        self.population = filtered_population


        fronts = self.map_elites.non_dominated_sort()
        self.fronts = fronts
        crowding_dists = self.map_elites.crowding_distance(fronts)
        self.crowding_dists = crowding_dists
        self.map_elites.select_new_population(fronts, crowding_dists)
        self.check_objectives()
    

    def generate_run_data(self):
        model = self.map_elites.next_model()
        return Work(model, self.score_strategy, self.game, self.stop_after, self.run_name_with_params, self.num_levels, self.objectives)

    def write_work_for_child(self, work, child_name):
        path = self.WORK_DIR / str('task-id-{}-child-{}'.format(self.work_per_generation, child_name))
        path = path.with_suffix('.pkl')
        logging.info('Writing work %s', str(path))
        pickle.dump(work, path.open('wb'))
