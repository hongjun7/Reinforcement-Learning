import copy
import random
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float)


class ReplayBuffer(object):
    def __init__(self, buffer_limit):
        self.buffer = []
        self.buffer_limit = buffer_limit
    
    def put(self, transition):
        if len(self.buffer) >= self.buffer_limit:
            self.buffer = self.buffer[self.buffer_limit//5:]
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, min(len(self.buffer), n))
        states, actions, rewards, next_states, masks = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            
            # states and actions are lists (no-wrap)
            states.append(s)
            actions.append(a)
            next_states.append(s_prime)
            
            # rewards and masks are float values (wrap with list type)
            rewards.append([r])
            masks.append([done])
        
        states = to_tensor(states)
        actions = to_tensor(actions)
        rewards = to_tensor(rewards)
        next_states = to_tensor(next_states)
        masks = to_tensor(masks)
        return states, actions, rewards, next_states, masks
    
    def size(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.L1 = nn.Linear(state_dim, 400)
        self.L2 = nn.Linear(400, 300)
        self.L3 = nn.Linear(300, action_dim)
    
    def forward(self, state):
        a = torch.nn.functional.relu(self.L1(state))
        a = torch.nn.functional.relu(self.L2(a))
        return torch.tanh(self.L3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.L1 = nn.Linear(state_dim + action_dim, 400)
        self.L2 = nn.Linear(400, 300)
        self.L3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        Q = torch.nn.functional.relu(self.L1(sa))
        Q = torch.nn.functional.relu(self.L2(Q))
        Q = self.L3(Q)
        return Q


class Model(object):
    def __init__(self, state_dim, action_dim,
                 lr=0.001, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_period=2):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.q1 = Critic(state_dim, action_dim).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)

        self.q2 = Critic(state_dim, action_dim).to(device)
        self.q2_target = copy.deepcopy(self.q2)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
    
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_period = policy_period
        
        self.n = 0
    
    def step(self, state):
        state = to_tensor(state.reshape(1, -1)).to(device)
        return self.actor.forward(state).cpu().data.numpy().flatten()
    
    def soft_update(self, net, net_target):
        for param, param_target in zip(net.parameters(), net_target.parameters()):
            param_target.data.copy_(
                param.data * self.tau + param_target.data * (1.0 - self.tau)
            )
    
    def train(self, memory, gradient_steps, batch_size):
        self.n += 1
        for epoch in range(gradient_steps):
            # sample replay buffer
            state, action, reward, next_state, done = memory.sample(batch_size)
        
            with torch.no_grad():
                # select action according to policy and add clipped noise
                noise = (
                        torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                next_action = self.actor_target.forward(next_state) + noise
                next_action = next_action.clamp(-1, +1)
                
                # compute the target Q value
                target_Q1 = self.q1_target.forward(next_state, next_action)
                target_Q2 = self.q2_target.forward(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()
            
            # get current Q estimates
            current_Q1 = self.q1.forward(state, action)
            current_Q2 = self.q2.forward(state, action)
            
            # compute critic loss
            q1_loss = torch.nn.functional.mse_loss(current_Q1, target_Q)
            q2_loss = torch.nn.functional.mse_loss(current_Q2, target_Q)
        
            # optimize the critic
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()
    
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()
        
            # delayed policy updates
            if epoch % self.policy_period == 0:
                # compute actor loss
                actor_loss = -self.q1.forward(state, self.actor.forward(state)).mean()
            
                # optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
            
                # update the frozen target models
                self.soft_update(self.q1, self.q1_target)
                self.soft_update(self.q2, self.q2_target)
                self.soft_update(self.actor, self.actor_target)
    
    def save(self):
        torch.save(self.actor.state_dict(), './weights/td3-BipedalWalker/actor.pth')
        torch.save(self.actor_target.state_dict(), './weights/td3-BipedalWalker/actor_target.pth')
        
        torch.save(self.q1.state_dict(), './weights/td3-BipedalWalker/q1.pth')
        torch.save(self.q1_target.state_dict(), './weights/td3-BipedalWalker/q1_target.pth')
        
        torch.save(self.q2.state_dict(), './weights/td3-BipedalWalker/q2.pth')
        torch.save(self.q2_target.state_dict(), './weights/td3-BipedalWalker/q2_target.pth')
    
    def load(self):
        self.actor.load_state_dict(torch.load('./weights/td3-BipedalWalker/actor.pth'))
        self.actor_target.load_state_dict(torch.load('./weights/td3-BipedalWalker/actor_target.pth'))
        
        self.q1.load_state_dict(torch.load('./weights/td3-BipedalWalker/q1.pth'))
        self.q1_target.load_state_dict(torch.load('./weights/td3-BipedalWalker/q1_target.pth'))

        self.q1.load_state_dict(torch.load('./weights/td3-BipedalWalker/q1.pth'))
        self.q1_target.load_state_dict(torch.load('./weights/td3-BipedalWalker/q1_target.pth'))
