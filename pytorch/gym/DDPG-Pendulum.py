import gym
import numpy as np
from pytorch.model import DDPG


class Noise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)
    
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


env = gym.make('Pendulum-v0')
memory = DDPG.ReplayBuffer(buffer_limit=50000)

model = DDPG.Model(lr_mu=0.0005, lr_q=0.001, gamma=0.99, batch_size=32, tau=0.005)

score = 0.0
print_interval = 20

ou_noise = Noise(mu=np.zeros(1))
scores = []
for n_epi in range(10000):
    s = env.reset()
    done = False
    
    while not done:
        a = model.step(s)
        a = a.item() + ou_noise()[0]
        s_prime, r, done, info = env.step([a])
        memory.put((s, a, r / 100.0, s_prime, done))
        score += r
        s = s_prime
    
    if memory.size() > 2000:
        for i in range(10):
            model.train(memory)
    
    if n_epi % print_interval == 0 and n_epi != 0:
        print("# of episode : {}, avg score : {:.1f}".format(n_epi, score / print_interval))
        scores.append(score)
        score = 0.0
env.close()
