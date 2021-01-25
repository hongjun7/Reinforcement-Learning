import gym
from tf2.model import PG

env = gym.make('CartPole-v0').unwrapped
env.seed(1)
episode_length = 1000

agent = PG.Model(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n, lr=0.01, gamma=0.99)

for episode in range(episode_length):
    obs_cur = env.reset()
    episode_reward = 0

    while True:
        action = agent.step(obs_cur)
        obs_nxt, reward, done, _ = env.step(action)

        agent.memory.store(obs_cur, action, reward)

        obs_cur = obs_nxt
        episode_reward += reward

        if done:
            agent.learn()
            print('episode: %i' % episode, ", reward: %i" % episode_reward)
            break
env.close()
