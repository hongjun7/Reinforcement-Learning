import gym
from tf2.model import DQN

env = gym.make('CartPole-v0').unwrapped
env.seed(1)
episode_length = 1000

agent = DQN.Model(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n, lr=0.01, gamma=0.999,
                  epsilon=0.2, buffer_capacity=int(1e6))
step = 0
epsilon_step = 10

for episode in range(episode_length):
    obs_cur = env.reset()
    episode_reward = 0

    while True:
        action = agent.step(obs_cur)
        obs_nxt, reward, done, _ = env.step(action)

        agent.memory.store_transition(obs_cur, action, reward, obs_nxt, done)

        obs_cur = obs_nxt
        episode_reward += reward
        
        if step >= 128 * 3:
            agent.learn()
            if step % epsilon_step == 0:
                agent.epsilon = max([agent.epsilon * 0.99, 0.001])
        
        if done:
            print('episode: %i' % episode, ", reward: %i" % episode_reward)
            break

        step += 1
env.close()
