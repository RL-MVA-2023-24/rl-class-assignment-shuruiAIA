from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import pickle

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


class ProjectAgent:

    def act(self, observation, use_random=False):
        network = self.model
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        
    def save(self, path):
        
        pass

    def load(self):
        
        self.model = torch.load("src/model.pt") 
        pass
