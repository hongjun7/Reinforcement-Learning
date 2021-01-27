import gym
from tf2.model import PPO

env = gym.make('LunarLander-v2').unwrapped
env.seed(1)

episode_length = 1000

agent = PPO.Model(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n,
                  lr=0.001, gamma=0.99, clip_range=0.2, update_ep_epochs=4)

for episode in range(episode_length):
    obs_cur = env.reset()
    episode_reward = 0
    
    step = 0
    while True:
        step += 1
        action, logprob = agent.step(obs_cur)
        obs_nxt, reward, ep_done, _ = env.step(action)
        if episode % 50 == 0:
            env.render()
        
        agent.memory.store(obs_cur, action, reward, logprob)

        obs_cur = obs_nxt
        episode_reward += reward

        if ep_done or step > 1000:
            agent.learn()
            print('episode: %i' % episode, ", step: %i" % step, ", reward: %i" % episode_reward)
            break
env.close()
