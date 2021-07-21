import random
from collections import deque

import gym
import torch
from pytorch.model import DQN

model = DQN.Model(lr=0.0005, gamma=0.99, target_interval=1000)

history = deque(maxlen=1000000)  # replay buffer

# gym environment
env = gym.make("CartPole-v0")
max_time_steps = 1000

# for computing average reward over 100 episodes
reward_history = deque(maxlen=100)
reward_list = []

# training
for episode in range(700):
    # sum of accumulated rewards
    rewards = 0

    # get initial observation
    observation = env.reset()
    state = torch.tensor(observation, dtype=torch.float32)

    # loop until an episode ends
    for t in range(1, max_time_steps + 1):
        # display current environment
        # epsilon greedy policy for current observation
        with torch.no_grad():
            if random.random() < 0.01:
                action = env.action_space.sample()
            else:
                action = torch.argmax(model.Q(state)).item()

        # get next observation and current reward for the chosen action
        observation_next, reward, done, info = env.step(action)
        state_next = torch.tensor(observation_next, dtype=torch.float32)

        # collect reward
        rewards = rewards + reward

        # collect a transition
        history.append([state, action, state_next, reward, done])

        model.learn(history=history)

        if done:
            break

        # pass observation to the next step
        observation = observation_next
        state = state_next

    # compute average reward
    reward_history.append(rewards)
    avg = sum(reward_history) / len(reward_history)
    reward_list.append(rewards)
    if (episode + 1) % 10 == 0:
        print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode + 1, rewards, avg))
env.close()
