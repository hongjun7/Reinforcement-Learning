import gym
import tensorflow as tf
from tf2.model import PPO

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

env = gym.make('LunarLander-v2').unwrapped
env.seed(1)

log_interval = 20
episode_length = 50000
update_timestep = 2000

agent = PPO.Model(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n,
                  lr=5e-4, gamma=0.99, clip_range=0.2, update_ep_epochs=4, vf_coef=0.5, ent_coef=1e-4)

step, running_reward, sum_episode_length = 0, 0, 0

for episode in range(1, episode_length+1):
    obs_cur = env.reset()
    ep_length = 0
    while True:
        step += 1
        ep_length += 1
        action, logprob = agent.step(obs_cur)
        obs_nxt, reward, ep_done, _ = env.step(action)
        
        agent.memory.store(obs_cur, action, reward, ep_done, logprob)

        obs_cur = obs_nxt
        running_reward += reward
        
        if step % update_timestep == 0:
            agent.learn()
            step = 0
        
        if ep_done or ep_length >= 300:
            break
    sum_episode_length += ep_length
    if episode % log_interval == 0:
        avg_length, running_reward = int(sum_episode_length / log_interval), int(running_reward / log_interval)
        print('Ep {} \t avg. length: {} \t avg. reward: {}'.format(episode, avg_length, running_reward))
        if running_reward >= 200:
            print('Solved LunarLander!')
            break
        sum_episode_length, running_reward = 0, 0
env.close()
