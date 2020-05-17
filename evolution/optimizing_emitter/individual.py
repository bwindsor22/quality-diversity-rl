class Individual:
    def __init__(self, feature, flattened, fitness):
        self.feature = feature
        self.flattened_state_dict = flattened
        self.fitness = fitness

    def __repr__(self):
        return '{} - {}'.format(self.feature, str(self.fitness))