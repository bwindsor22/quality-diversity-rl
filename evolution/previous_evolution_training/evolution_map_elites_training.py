import random

from models.train_dqn import evaluate_net


class ParamsAgent:
    def __init__(self, BATCH_SIZE, LINEAR_INPUT_SCALAR, KERNEL_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.LINEAR_INPUT_SCALAR = LINEAR_INPUT_SCALAR
        self.KERNEL_SIZE = KERNEL_SIZE

    def __repr__(self):
        return f'batch_{self.BATCH_SIZE}-linear_{self.LINEAR_INPUT_SCALAR}-kernel_{self.KERNEL_SIZE}'

    def feature_descriptor(self):
        kernel_bucket = 10 if self.KERNEL_SIZE >= 10 else 5 if self.KERNEL_SIZE >= 5 else 1
        scalar_bucket = 12 if self.LINEAR_INPUT_SCALAR >= 12 else 6 if self.LINEAR_INPUT_SCALAR >= 6 else 1
        batch_bucket = 400 if self.BATCH_SIZE >= 400 else 200 if self.BATCH_SIZE >= 200 else 1
        return f'{kernel_bucket}-{scalar_bucket}-{batch_bucket}'


class GeneticEvolver:
    def __init__(self):
        self.SAMPLED_POPULATION_SIZE = 5 # 8
        self.GENERATIONS = 10000 # 5
        self.MUTATION_PROBABILITY = 0.2

        self.DEFAULT_BATCH_SIZE = 128
        self.DEFAULT_LINEAR_INPUT_SCALAR = 8
        self.DEFAULT_KERNEL_SIZE = 3

        # feature_descriptor to max_score
        self.performances = {}

        # feature descriptor to agent
        self.solutions = {}

    def run(self):
        file = open("evol_training.txt", "w+")
        file.write('\n\n\n begin new run \n\n\n')
        file.flush()

        initial_agents = [
            self.get_inital_agent()
            for _ in range(self.SAMPLED_POPULATION_SIZE)]

        # load initial agents to performance and solutions arrays
        for agent in initial_agents:
            feat_descriptor = agent.feature_descriptor()
            try:
                fitness_score = self.eval_agent(agent, file)
                if not feat_descriptor in self.performances or self.performances[feat_descriptor] < fitness_score:
                    file.write('\n Agent exceeded max score' + str(feat_descriptor))
                    self.performances[feat_descriptor] = fitness_score
                    self.solutions[feat_descriptor] = agent
            except Exception as e:
                print('Invalid agent')
                print(e)

        for p_iter in range(self.GENERATIONS):
            file.write('\nBeginning iter' + str(p_iter))
            file.flush()
            try:

                agent = self.get_new_random_variation_agent()

                feat_descriptor = agent.feature_descriptor()
                fitness_score = self.eval_agent(agent, file)
                file.write('\nNew agent' + str(initial_agents) + ' | score ' + str(fitness_score))

                if not feat_descriptor in self.performances or self.performances[feat_descriptor] < fitness_score:
                    file.write('\nAgent exceeded max score' + str(feat_descriptor))
                    self.performances[feat_descriptor] = fitness_score
                    self.solutions[feat_descriptor] = agent

                if p_iter % 3:
                    file.write('\n\n REACHED {} iterations'.format(p_iter))
                    file.write('\nperformances')
                    file.write(str(self.performances))
                    file.write('\nagents ')
                    file.write(str(self.solutions))

            except Exception as e:
                print('Invalid agent during training')
                print(e)

    def get_inital_agent(self):
        batch = int(2 * random.random() * self.DEFAULT_BATCH_SIZE) + 1
        linear = int(2 * random.random() * self.DEFAULT_LINEAR_INPUT_SCALAR) + 1
        kernel = min(int(1.5 * random.random() * self.DEFAULT_KERNEL_SIZE) + 1, 10)
        return ParamsAgent(batch, linear, kernel)


    def eval_agent(self, agent, file):
        file.write('\nEVALUATING AGENT' + str(agent))
        score_after_training = evaluate_net(BATCH_SIZE=agent.BATCH_SIZE,
                                            LINEAR_INPUT_SCALAR=agent.LINEAR_INPUT_SCALAR,
                                            KERNEL=agent.KERNEL_SIZE)
        return score_after_training


    def get_new_random_variation_agent(self):
        parent1, parent2 = self.get_sample_agents()

        child = self.cross_breed(parent1, parent2)

        # mutate child
        if random.random() < self.MUTATION_PROBABILITY:
            batch = int(2 * random.random() * child.BATCH_SIZE) + 1
            linear = int(2 * random.random() * child.LINEAR_INPUT_SCALAR) + 1
            kernel = min(int(2 * random.random() * child.KERNEL_SIZE) + 1, 10)
            return ParamsAgent(BATCH_SIZE=batch, LINEAR_INPUT_SCALAR=linear, KERNEL_SIZE=kernel)
        return child

    def cross_breed(self, parent1, parent2):
        rand = random.random()
        if rand < 0.16:
            return ParamsAgent(BATCH_SIZE=parent1.BATCH_SIZE,
                               LINEAR_INPUT_SCALAR=parent1.LINEAR_INPUT_SCALAR,
                               KERNEL_SIZE=parent2.KERNEL_SIZE)

        elif rand < 0.32:
            return ParamsAgent(BATCH_SIZE=parent1.BATCH_SIZE,
                               LINEAR_INPUT_SCALAR=parent2.LINEAR_INPUT_SCALAR,
                               KERNEL_SIZE=parent1.KERNEL_SIZE)
        elif rand < 0.48:
            return ParamsAgent(BATCH_SIZE=parent1.BATCH_SIZE,
                               LINEAR_INPUT_SCALAR=parent2.LINEAR_INPUT_SCALAR,
                               KERNEL_SIZE=parent2.KERNEL_SIZE)
        elif rand < 0.64:
            return ParamsAgent(BATCH_SIZE=parent2.BATCH_SIZE,
                               LINEAR_INPUT_SCALAR=parent1.LINEAR_INPUT_SCALAR,
                               KERNEL_SIZE=parent1.KERNEL_SIZE)
        elif rand < 0.80:
            return ParamsAgent(BATCH_SIZE=parent2.BATCH_SIZE,
                               LINEAR_INPUT_SCALAR=parent2.LINEAR_INPUT_SCALAR,
                               KERNEL_SIZE=parent1.KERNEL_SIZE)
        else:
            return ParamsAgent(BATCH_SIZE=parent2.BATCH_SIZE,
                               LINEAR_INPUT_SCALAR=parent1.LINEAR_INPUT_SCALAR,
                               KERNEL_SIZE=parent2.KERNEL_SIZE)


    def get_sample_agents(self):
        return random.sample(list(self.solutions.values()), 2)


if __name__ == '__main__':
    GeneticEvolver().run()