class Work:
    def __init__(self, model, score_strategy,game,stop_after,run_name):
        self.model = model
        self.score_strategy = score_strategy
        self.game = game
        self.stop_after = stop_after
        self.run_name = run_name