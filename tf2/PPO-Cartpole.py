import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers


class Memory:
    def __init__(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

    def store(self, obs, act, rwd):
        self.ep_obs.append(obs)
        self.ep_act.append(act)
        self.ep_rwd.append(rwd)

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []


class ActorNetwork(object):
    def __init__(self, obs_dim, act_dim, lr):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.model = keras.Sequential(
            [
                layers.Dense(obs_dim, activation=tf.nn.tanh,
                             kernel_initializer=keras.initializers.glorot_uniform()),
                layers.Dense(10, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3)),
                layers.Dense(act_dim)
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=lr)

    def step(self, obs):
        return self.model(obs)

    def choose_action(self, obs):
        act_prob = self.step(obs)
        all_act_prob = tf.nn.softmax(act_prob)
        return all_act_prob

    def get_cross_entropy(self, obs, act):
        act_prob = self.step(obs)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act_prob, labels=act)


class ValueNetwork(object):
    def __init__(self, obs_dim, lr):
        self.model = keras.Sequential(
            [
                layers.Dense(obs_dim, activation=tf.nn.tanh,
                             kernel_initializer=keras.initializers.glorot_uniform()),
                layers.Dense(10, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3)),
                layers.Dense(1)
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=lr)

    def step(self, obs):
        return self.model(obs)


class PPO(object):
    def __init__(self, obs_dim, act_dim, lr_actor, lr_critic, gamma, clip_range, update_ep_epochs):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.clip_range = clip_range
        self.update_ep_epochs = update_ep_epochs

        self.actor = ActorNetwork(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n, lr=lr_actor)
        self.critic = ValueNetwork(obs_dim=env.observation_space.shape[0], lr=lr_critic)
        self.memory = Memory()

    def step(self, obs):
        if obs.ndim < 2:
            obs = obs[np.newaxis, :]
        prob_weights = self.actor.choose_action(obs).numpy()
        act = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        val = self.critic.step(obs)
        return act, val

    def learn(self, last_value, done):
        obs = np.vstack(self.memory.ep_obs)
        act = np.array(self.memory.ep_act)
        rwd = np.array(self.memory.ep_rwd)
        old_pi = self.actor.get_cross_entropy(obs, act)
        q_value = self.compute_q_value(last_value, done, rwd)

        for epoch in range(self.update_ep_epochs):
            with tf.GradientTape(persistent=True) as tape:
                advantage = (q_value - self.critic.step(obs))
                critic_loss = tf.reduce_mean(tf.square(advantage))
                critic_grad = tape.gradient(critic_loss, self.critic.model.trainable_variables)
                self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.model.trainable_variables))

                new_pi = self.actor.get_cross_entropy(obs, act)
                # subtracting the logs is equal to dividing the values and then canceling the log with exp.
                ratio = tf.exp(new_pi - old_pi)
                clip_ratio = tf.clip_by_value(ratio, 1. - self.clip_range, 1. + self.clip_range)
                actor_loss = tf.reduce_mean(tf.minimum(clip_ratio, ratio) * advantage)
                actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.model.trainable_variables))

        self.memory.reset()

    def compute_q_value(self, last_value, done, rwd):
        q_value = np.zeros_like(rwd)
        v = 0 if done else last_value
        for t in reversed(range(len(rwd))):
            v = v * self.gamma + rwd[t]
            q_value[t] = v
        return q_value[:, np.newaxis]


env = gym.make('CartPole-v0').unwrapped
env.seed(1)
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

episode_length = 1000

agent = PPO(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n,
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
