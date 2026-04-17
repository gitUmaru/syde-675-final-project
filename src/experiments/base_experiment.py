from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def run(self):
        pass


Experiment = BaseExperiment
