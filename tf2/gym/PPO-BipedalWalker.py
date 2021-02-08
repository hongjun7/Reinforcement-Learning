import gym
import tensorflow as tf
from tf2.model import PPO2

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

env = gym.make('BipedalWalker-v3').unwrapped
env.seed(1)

log_interval = 10
episode_length = 50000
update_timestep = 4000

agent = PPO2.Model(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0],
                   lr=3e-4, gamma=0.99, clip_range=0.2, update_ep_epochs=10, vf_coef=0.5, ent_coef=1e-4)

step, running_reward, sum_episode_length = 0, 0, 0

solved = False
agent.load()
for episode in range(0, episode_length):
    obs_cur = env.reset()
    ep_length = 0
    while True:
        step += 1
        ep_length += 1
        action, logprob = agent.step(obs_cur)
        obs_nxt, reward, ep_done, _ = env.step(action)
        env.render()
        agent.memory.store(obs_cur, action, reward, ep_done, logprob)
        
        obs_cur = obs_nxt
        running_reward += reward
        
        if step % update_timestep == 0:
            agent.learn()
            step = 0
        
        if ep_done or ep_length >= 1000:
            break
    sum_episode_length += ep_length
    if episode % log_interval == 0:
        avg_length, running_reward = int(sum_episode_length / log_interval), int(running_reward / log_interval)
        print('Ep {} \t avg. length: {} \t avg. reward: {}'.format(episode, avg_length, running_reward))
        if running_reward >= 300:
            solved = True
        if episode % 200 == 0:
            agent.save()
        sum_episode_length, running_reward = 0, 0
env.close()
