class History:
    def __init__(self):
        self.history = {}

    def clean(self):
        self.history = {}

    def add(self, name, observation):
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(observation)

    def __getitem__(self, item):
        return self.history[item]
