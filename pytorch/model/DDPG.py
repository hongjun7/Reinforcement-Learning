import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def f_tensor(x):
    return torch.tensor(x, dtype=torch.float)


class ReplayBuffer(object):
    def __init__(self, buffer_limit=50000):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])
        
        return f_tensor(s_lst), f_tensor(a_lst), f_tensor(r_lst), f_tensor(s_prime_lst), f_tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # [-1, 1] → action space [-2, 2]
        mu = torch.tanh(self.fc_mu(x)) * 2
        return mu


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
    
    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class Model(object):
    def __init__(self, lr_mu, lr_q, gamma, batch_size, tau):
        self.lr_mu = lr_mu
        self.lr_q = lr_q
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        
        self.q, self.q_target = QNet(), QNet()
        self.q_target.load_state_dict(self.q.state_dict())
        self.mu, self.mu_target = MuNet(), MuNet()
        self.mu_target.load_state_dict(self.mu.state_dict())
        
        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=lr_mu)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=lr_q)
    
    def step(self, state):
        return self.mu(torch.from_numpy(state).float())
    
    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
    
    def train(self, memory):
        s, a, r, s_prime, done_mask = memory.sample(self.batch_size)

        target = r + self.gamma * self.q_target.forward(s_prime, self.mu_target(s_prime)) * done_mask
        critic_loss = F.mse_loss(target.detach(), self.q.forward(s, a))

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()
        
        actor_loss = -self.q.forward(s, self.mu(s)).mean()
        
        self.mu_optimizer.zero_grad()
        actor_loss.backward()
        self.mu_optimizer.step()
        
        self.soft_update(self.mu, self.mu_target)
        self.soft_update(self.q, self.q_target)
