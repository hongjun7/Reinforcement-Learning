import gym
import numpy as np
import tensorflow as tf
from tf2.model import TD3


for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

env = gym.make('BipedalWalker-v3')
memory = TD3.ReplayBuffer(buffer_limit=int(5e5))

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
model = TD3.Model(state_dim=state_dim, action_dim=action_dim,
                  lr=0.001, gamma=0.99, tau=0.005,
                  policy_noise=0.2, noise_clip=0.5, policy_period=2)
print_interval = 10
batch_size = 100

scores = []
for epoch in range(5000):
    state = env.reset()
    done = False
    
    t, score = 0, 0
    while not done:
        action = model.step(state)
        action = action + np.random.normal(0, 0.1, size=env.action_space.shape[0])
        action = action.clip(env.action_space.low, env.action_space.high)
        
        next_state, reward, done, _ = env.step(action)
        memory.put((state, action, reward, next_state, done))
        score += reward
        state = next_state
        t += 1
    scores.append(score)
    
    model.train(memory, t, batch_size)
    
    if (epoch + 1) % print_interval == 0:
        moving_avg_score = sum(scores[-print_interval:]) / print_interval
        print("#{} episodes, avg score : {:.1f}".format(epoch + 1, moving_avg_score))
env.close()
