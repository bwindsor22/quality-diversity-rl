import random

from models.train_dqn import run_training_for_params


class ParamsAgent:
    def __init__(self, BATCH_SIZE, LINEAR_INPUT_SCALAR):
        self.BATCH_SIZE = BATCH_SIZE
        self.LINEAR_INPUT_SCALAR = LINEAR_INPUT_SCALAR

    def __repr__(self):
        return f'batch_{self.BATCH_SIZE}-linear_{self.LINEAR_INPUT_SCALAR}'


class GeneticEvolver:
    def __init__(self):
        self.POPULATION_SIZE = 8 # 8
        self.GENERATIONS = 100 # 5
        self.MUTATION_PROBABILITY = 0.1
        self.PARENT_PASS_ON_PROBABILITY = 0.7

        self.DEFAULT_BATCH_SIZE = 128
        self.DEFAULT_LINEAR_INPUT_SCALAR = 8

    def run(self):
        random_agents = [
            self.get_inital_agent()
            for _ in range(self.POPULATION_SIZE)]
        initial_agents = random_agents
        file = open("evol_training.txt", "w+")

        # try:
        for p_iter in range(self.GENERATIONS):
            file.write('\nBeginning generation' + str(p_iter))
            file.write('\nAgents' + str(initial_agents))
            ranked_agents = self.rank_agents(initial_agents, file)
            file.write('\nranked agents' + str(ranked_agents))
            range_lookup = self.generate_rank_selection_lookup([ranked[0] for ranked in ranked_agents])
            file.write('\nGenerating new population of agents for next round')
            new_population = self.get_new_population(range_lookup)
            initial_agents = self.mutate_population(new_population)

        file.write('\ndoing final eval')
        ranked_sequences = self.rank_agents(initial_agents)
        file.write('\nranked agents')
        file.write(str(ranked_sequences))
        # except Exception as e:

    def get_inital_agent(self):
        batch = int(2 * random.random() * self.DEFAULT_BATCH_SIZE) + 1
        linear = int(2 * random.random() * self.DEFAULT_LINEAR_INPUT_SCALAR) + 1
        return ParamsAgent(batch, linear)

    def rank_agents(self, agents, file):
        """
        rank agents with the best first
        """
        scored_agents = []
        for agent in agents:
            scored_agents.append(
                (agent, self.eval_agent(agent, file))
            )
        agents_sorted = sorted(scored_agents, key=lambda x: x[1], reverse=True)
        return agents_sorted

    def eval_agent(self, agent, file):
        file.write('\nEVALUATING AGENT' + str(agent))
        score_after_training = run_training_for_params(BATCH_SIZE=agent.BATCH_SIZE,
                                                       LINEAR_INPUT_SCALAR=agent.LINEAR_INPUT_SCALAR)
        return score_after_training

    def generate_rank_selection_lookup(self, ranked_agents):
        """
        Creating a lookup dict for agent selection.

        Using ranked agents, generate list of dicts such that
        if random.random() is below the threshold, the sequence should be chosen
        [
            {
                threshold_probability
                agent
            }
        ]
        """
        total = float(sum([i + 1 for i in range(self.POPULATION_SIZE)]))
        range_lookup = []
        prev_threshold = 0
        for i, agent in enumerate(ranked_agents):
            rank = self.POPULATION_SIZE - i
            prob = rank / total
            threshold = prob + prev_threshold
            range_lookup.append({
                'agent': agent,
                'threshold_probability': threshold
            })
            prev_threshold = threshold
        return range_lookup

    def get_new_population(self, range_lookup):
        new_population = []
        while len(new_population) < self.POPULATION_SIZE:
            parent1 = self.get_sample_agent(range_lookup)
            parent2 = self.get_sample_agent(range_lookup)

            if random.random() >= self.PARENT_PASS_ON_PROBABILITY:
                new_population.append(parent1)
                if len(new_population) < self.POPULATION_SIZE:
                    new_population.append(parent2)
            else:
                new_population.append(self.cross_breed(parent1, parent2))
        return new_population

    def cross_breed(self, parent1, parent2):
        if random.random() < 0.5:
            return ParamsAgent(BATCH_SIZE=parent1.BATCH_SIZE, LINEAR_INPUT_SCALAR=parent2.LINEAR_INPUT_SCALAR)
        else:
            return ParamsAgent(BATCH_SIZE=parent2.BATCH_SIZE, LINEAR_INPUT_SCALAR=parent1.LINEAR_INPUT_SCALAR)

    def get_sample_agent(self, range_lookup):
        choice = random.random()
        for option in range_lookup:
            if choice <= option['threshold_probability']:
                return option['agent']


    def mutate_population(self, agents):
        """
        After you get all the new chromosomes of the new population, apply a random test for each chromosome.
        if the test result is less (or equal) to 10%, mutate the chromosome by random choice. Choose a random action
        in the chromosome sequence and replace it by another action, chose it randomly.
        """
        print('mutating agents')
        for i in range(self.POPULATION_SIZE):
            if random.random() < self.MUTATION_PROBABILITY:
                agent = agents[i]
                batch = int(2 * random.random() * agent.BATCH_SIZE) + 1
                linear = int(2 * random.random() * agent.LINEAR_INPUT_SCALAR) + 1
                agents[i] = ParamsAgent(BATCH_SIZE=batch, LINEAR_INPUT_SCALAR=linear)
        return agents


if __name__ == '__main__':
    GeneticEvolver().run()