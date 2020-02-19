
import sys
from random import randint
import gym
from time import time
from gym.wrappers import Monitor
from collections import deque


# A generic game evaluator.
# Make specific evaluators if feature info is
# required to be recorded and stored.
class GameEvaluator:
    def __init__(self, game_name, seed=1009, num_rep=1):
        self.env = gym.make(game_name)
        self.seed = seed
        self.num_rep = num_rep
        self.num_actions = self.env.action_space.n
        print(self.num_actions)

    def run(self, agent, render=False):
        agent.fitness = 0
        self.env.seed(self.seed)
        env = self.env
        if render:
            env = Monitor(env, './videos/'+str(time())+'/')
        observation = env.reset()

        action_frequency = [0] * self.num_actions
        
        action_count = 0
        done = False
        while not done:
            #if render:
            #    env.render()
            
            pos = min(action_count//self.num_rep, len(agent.commands)-1)
            action = agent.commands[pos]
            action_count += 1

            observation, reward, done, info = env.step(action)
            agent.fitness += reward

            action_frequency[action] += 1
        
        final_observation = list(observation)

        # calculate RLE approximation 
        #numNewChars = 0
        #prevChar = -2
        #for cmd in agent.commands:
        #    if cmd != prevChar:
        #        numNewChars = numNewChars + 1
        #        prevChar = cmd

        # calculate polynomial hash
        #b1 = 3
        #b2 = 7

        #runningHash1 = 0
        #runningHash2 = 0
        #for cmd in agent.commands:
            #runningHash1 = (runningHash1 * b1 + cmd) % len(agent.commands)
            #runningHash2 = (runningHash2 * b2 + cmd) % len(agent.commands)

        #agent.features = tuple(final_observation[:1])
        #agent.features = (numNewChars, numNewChars)
        #agent.features = (runningHash1, runningHash2)
        agent.features = (agent.fitness, agent.fitness)
        agent.action_count = action_count
        
class Agent:
    
    def __init__(self, game, sequence_len):
        self.fitness = 0
        self.game = game
        self.sequence_len = sequence_len
        self.commands = [char for char in str(2211221010102033021232100200302221032322301110220222120113321130212300001110022131020031331113221131)]
        #self.commands = [
            #randint(0, game.num_actions-1) for _ in range(sequence_len)
        #]

    def mutate(self):
        child = Agent(self.game, self.sequence_len)
        i = randint(0, self.sequence_len-1)
        offset = randint(1, self.game.num_actions)
        child.commands[i] = \
            (child.commands[i] + offset) % self.game.num_actions
        return child

class LinearSizer:
    def __init__(self, start_size, end_size):
        self.min_size = start_size
        self.range = end_size-start_size

    def get_size(self, portion_done):
        size = int((portion_done+1e-9)*self.range) + self.min_size
        return min(size, self.min_size+self.range)

class ExponentialSizer:
    def __init__(self, start_size, end_size):
        self.min_size = start_size
        self.max_size = end_size

    def get_size(self, portion_done):
        cur_size = self.max_size
        while portion_done < 0.5 and cur_size > self.min_size:
            cur_size //= 2
            portion_done *= 2

        return cur_size


class EmptyBuffer:

    def is_overpopulated(self):
        return False

    def add_individual(self, to_add):
        pass

    def remove_individual(self):
        return None

class SlidingBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer_queue = deque(maxlen=buffer_size+1)

    def is_overpopulated(self):
        return len(self.buffer_queue) > self.buffer_size

    def add_individual(self, to_add):
        self.buffer_queue.append(to_add)

    def remove_individual(self):
        return self.buffer_queue.popleft()
      
def runRS(runnum, game, sequence_len, num_individuals):
    best_fitness = -10 ** 18
    best_sequence = None
    whenfound = 0

    for agent_id in range(num_individuals):
        agent = Agent(game, sequence_len)
        game.run(agent)
        if agent.fitness > best_fitness:
            best_fitness = agent.fitness
            best_sequence = agent.commands
            whenfound = agent_id
            game.run(agent, render=False)
        if agent_id % 100 == 0:
            print(agent_id, best_fitness)

    with open('results' + str(runnum) + '.txt', 'a') as f:
        f.write(str(whenfound) + " " + str(best_fitness) + "\n")

    return best_fitness, best_sequence      
      
      
           
def runES(runnum, game, sequence_len, is_plus=False, 
        num_parents=None, population_size=None, 
        num_generations=None):

    best_fitness = -10 ** 18
    best_sequence = None
    whenfound = 0

    population = [Agent(game, sequence_len) for _ in range(population_size)]
    for p in population:
        game.run(p)
        if p.fitness > best_fitness:
            best_fitness = p.fitness
            best_sequence = p.commands
    
    print(best_fitness)

    for curGen in range(num_generations):
        population.sort(reverse=True, key=lambda p: p.fitness)
        parents = population[:num_parents]

        population = []
        for i in range(population_size):
            p = parents[randint(0, len(parents)-1)]
            child = p.mutate()
            game.run(child)

            if child.fitness > best_fitness:
                best_fitness = child.fitness
                best_sequence = child.commands
                whenfound = curGen*population_size + i
                game.run(child, render=False)
            population.append(child)
        
        print(curGen, parents[0].fitness, best_fitness)

        if is_plus:
            population += parents

    with open('results' + str(runnum) + '.txt', 'a') as f:
        f.write(str(whenfound) + " " + str(best_fitness) + "\n")

    return best_fitness, best_sequence
  
class FixedFeatureMap:
    
    def __init__(self, num_to_evaluate, buffer_size, boundaries, sizer):

        # Clock for resizing the map.
        self.num_individuals_to_evaluate = num_to_evaluate
        self.num_individuals_added = 0
        
        # Feature to individual mapping.
        self.num_features = len(boundaries)
        self.boundaries = boundaries
        self.elite_map = {}
        self.elite_indices = []

        # A group is the number of cells along 
        # each dimension in the feature space.
        self.group_sizer = sizer
        self.num_groups = 3

        if buffer_size == None:
            self.buffer = EmptyBuffer()
        else:
            self.buffer = SlidingBuffer(buffer_size)

    def get_feature_index(self, feature_id, feature):
        low_bound, high_bound = self.boundaries[feature_id]
        if feature <= low_bound:
            return 0
        if high_bound <= feature:
            return self.num_groups-1

        gap = high_bound - low_bound + 1
        pos = feature - low_bound
        index = int(self.num_groups * pos / gap)
        return index

    def get_index(self, agent):
        index = tuple(self.get_feature_index(i, v) \
                for i, v in enumerate(agent.features))
        return index

    def add_to_map(self, to_add):
        index = self.get_index(to_add)

        replaced_elite = False
        if index not in self.elite_map:
            self.elite_indices.append(index)
            self.elite_map[index] = to_add
            replaced_elite = True
        elif self.elite_map[index].fitness < to_add.fitness:
            self.elite_map[index] = to_add
            replaced_elite = True

        return replaced_elite

    def remove_from_map(self, to_remove):
        index = self.get_index(to_remove)
        if index in self.elite_map and self.elite_map[index] == to_remove:
            del self.elite_map[index]
            self.elite_indices.remove(index)
            return True

        return False

    def remap(self, next_num_groups):
        print('remap', '{}x{}'.format(next_num_groups, next_num_groups))
        self.num_groups = next_num_groups

        all_elites = self.elite_map.values()
        self.elite_indices = []
        self.elite_map = {}
        for elite in all_elites:
            self.add_to_map(elite)
        
    def add(self, to_add):
        self.num_individuals_added += 1
        portion_done = \
            self.num_individuals_added / self.num_individuals_to_evaluate
        next_num_groups = self.group_sizer.get_size(portion_done)
        if next_num_groups != self.num_groups:
            self.remap(next_num_groups)

        replaced_elite = self.add_to_map(to_add)
        self.buffer.add_individual(to_add)
        if self.buffer.is_overpopulated():
            self.remove_from_map(self.buffer.remove_individual())

        return replaced_elite

    def get_random_elite(self):
        pos = randint(0, len(self.elite_indices)-1)
        index = self.elite_indices[pos]
        return self.elite_map[index]

# For testing to make sure that the map works
#if __name__ == '__main__':
#    linear_sizer = LinearSizer(2, 10)
#    linear_sizer = ExponentialSizer(2, 500)
#    feature_map = FixedFeatureMap(100, None, [(0, 10), (0, 10)], linear_sizer)
#    print(feature_map.num_individuals_to_evaluate)

    #linear_sizer = ExponentialSizer(2, 500)
    #feature_map = FixedFeatureMap(500, 10, [(0, 10), (0, 10)], linear_sizer)
    #game = GameEvaluator('LunarLander-v2')

    #for x in range(0, 100):
    #    agent = Agent(game, 200)
    #    agent.features = (x%10, (x+5)%10)
    #    agent.fitness = -x
        #print(x, feature_map.add(agent))
    #    feature_map.add(agent)

  
def runME(runnum, game, sequence_len, 
        init_pop_size=-1, num_individuals=-1, sizer_type='Linear',
        sizer_range=(10,10), buffer_size=None, mortality=False):

    best_fitness = -10 ** 18
    best_sequence = None
    whenfound = 0

    sizer = None
    if sizer_type == 'Linear':
        sizer = LinearSizer(*sizer_range)
    elif sizer_type == 'Exponential':
        sizer = ExponentialSizer(*sizer_range)

    #feature_ranges = [(0, sequence_len)] * 2
    #feature_ranges = [(-1.0, 1.0), (0.0, 1.0)]
    #feature_ranges = [(0.0, sequence_len), (0.0, sequence_len)]
    feature_ranges = [(0.0, 200.0), (0.0, 200.0)]
    feature_ranges = feature_ranges[:2]
    #print(feature_ranges)
    feature_map = FixedFeatureMap(num_individuals, buffer_size,
                                  feature_ranges, sizer)

    for individuals_evaluated in range(num_individuals):

        cur_agent = None
        if individuals_evaluated < init_pop_size:
            cur_agent = Agent(game, sequence_len)
        else:
            cur_agent = feature_map.get_random_elite().mutate()

        game.run(cur_agent)
        feature_map.add(cur_agent)
        
        if cur_agent.fitness > best_fitness:
            print('improved:', cur_agent.fitness, cur_agent.action_count)
            best_fitness = cur_agent.fitness
            best_sequence = cur_agent.commands
            whenfound = individuals_evaluated
            game.run(cur_agent, render=False)

        #if individuals_evaluated % 1000 == 0:
            #elites = [feature_map.elite_map[index] for index in feature_map.elite_map]
            #indicies = [index for index in feature_map.elite_map]
            #features = list(zip(*[a.features for a in elites]))
            #for f in features:
            #    print(sorted(f))
            #print(indicies)
            

            #print(individuals_evaluated, best_fitness,
                  #len(feature_map.elite_indices))

    with open('results' + str(runnum) + '.txt', 'a') as f:
        f.write(str(whenfound) + " " + str(best_fitness) + "\n")
        for command in best_sequence:
                f.write(str(command))
        f.write("\n")

    return best_fitness, best_sequence


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    run = 0

    num_actions = 100
    search_type = 'test'
    #game = GameEvaluator('Qbert-v0', seed=1009, num_rep=2)
    game = GameEvaluator('LunarLander-v2', seed=1500, num_rep=3)

    if search_type == 'ES':
        runES(run, game, 
                num_actions, 
                is_plus=True,
                num_parents=10, 
                population_size=100,
                num_generations=1000,
            )
    elif search_type == 'RS':
        runRS(run, game, num_actions, 100000)
    elif search_type == 'ME':
        runME(run, game, 
                num_actions, 
                init_pop_size=1000, 
                num_individuals=100000, 
                sizer_type='Linear', 
                sizer_range=(200, 200), 
                buffer_size=5000, 
                mortality=True)
    elif search_type == 'test':
        #from gymjam.search import Agent
        cur_agent = Agent(game, num_actions)
        while True:
            game.run(cur_agent, render=True)

    game.env.close()

if __name__ == '__main__':
    sys.exit(main())