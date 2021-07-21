import random
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(4, 48)
        self.fcQ1 = Linear(48, 64)
        self.fcQ2 = Linear(64, 2)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.fcQ1(x)
        x = F.relu(x)
        x = self.fcQ2(x)
        return x


class Model(object):
    def __init__(self, lr, gamma, target_interval):
        self.gamma = gamma
        self.counter = 0
        self.target_interval = target_interval
        self.Q = MLP()
        self.Q_target = MLP()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
    
    def learn(self, history):
        if len(history) < 64:
            return
        batch = random.sample(history, 32)
        loss = torch.tensor(0, dtype=torch.float32)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        for state, action, state_next, reward, done in batch:
            q_value = self.Q.forward(state)[action]
            target = reward + self.gamma * torch.max(self.Q_target.forward(state_next)) * (1 - done)
            loss += loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.counter += 1
        if self.counter % self.target_interval == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
