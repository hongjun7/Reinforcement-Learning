import gym
from tf2.model import PPO

env = gym.make('CartPole-v0').unwrapped
env.seed(1)

episode_length = 1000

agent = PPO.Model(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n,
                  lr_actor=0.001, lr_critic=0.002, gamma=0.99, clip_range=0.2, update_ep_epochs=10)

for episode in range(episode_length):
    obs_cur = env.reset()
    episode_reward = 0

    while True:
        action, _ = agent.step(obs_cur)
        obs_nxt, reward, ep_done, _ = env.step(action)

        agent.memory.store(obs_cur, action, reward)

        obs_cur = obs_nxt
        episode_reward += reward

        if ep_done:
            _, val_cur = agent.step(obs_cur)
            agent.learn(val_cur, ep_done)
            print('episode: %i' % episode, ", reward: %i" % episode_reward)
            break
env.close()
