from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import torch
import torch.nn as nn

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

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
        self.model = QNetwork(6, 4, 256)
        checkpoint = torch.load("src/model.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.load_state_dict(checkpoint) 
        pass
