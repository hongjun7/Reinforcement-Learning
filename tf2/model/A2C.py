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
                layers.Dense(obs_dim),
                layers.Dense(10, activation="tanh"),
                layers.Dense(act_dim)
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=lr)

    def step(self, obs):
        return self.model(obs)

    def choose_action(self, obs):
        logit = self.step(obs)
        prob = tf.nn.softmax(logit).numpy()
        return np.random.choice(range(prob.shape[1]), p=prob.ravel())

    def get_cross_entropy(self, obs, act):
        logit = self.step(obs)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=act)


class ValueNetwork(object):
    def __init__(self, obs_dim, lr):
        self.model = keras.Sequential(
            [
                layers.Dense(obs_dim),
                layers.Dense(10, activation="relu", kernel_initializer=keras.initializers.he_uniform()),
                layers.Dense(1)
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=lr)

    def step(self, obs):
        return self.model(obs)


class Model(object):
    def __init__(self, obs_dim, act_dim, lr, gamma):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma

        self.actor = ActorNetwork(obs_dim=obs_dim, act_dim=act_dim, lr=self.lr)
        self.critic = ValueNetwork(obs_dim=obs_dim, lr=self.lr)
        self.memory = Memory()

    def step(self, obs):
        if obs.ndim < 2:
            obs = obs[np.newaxis, :]
        act = self.actor.choose_action(obs)
        val = self.critic.step(obs)
        return act, val

    def learn(self, last_value, done):
        obs = np.vstack(self.memory.ep_obs)
        act = np.array(self.memory.ep_act)
        rwd = np.array(self.memory.ep_rwd)
        q_value = self.compute_q_value(last_value, done, rwd)

        with tf.GradientTape() as tape:
            advantage = (q_value - self.critic.step(obs))
            cross_entropy = self.actor.get_cross_entropy(obs, act)
            actor_loss = tf.reduce_mean(cross_entropy * advantage)
            actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.model.trainable_variables))

        with tf.GradientTape() as tape:
            advantage = (q_value - self.critic.step(obs))
            critic_loss = tf.reduce_mean(tf.square(advantage))
            critic_grad = tape.gradient(critic_loss, self.critic.model.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.model.trainable_variables))

        self.memory.reset()

    def compute_q_value(self, last_value, done, rwd):
        q_value = np.zeros_like(rwd)
        v = 0 if done else last_value
        for t in reversed(range(len(rwd))):
            v = v * self.gamma + rwd[t]
            q_value[t] = v
        return q_value[:, np.newaxis]
