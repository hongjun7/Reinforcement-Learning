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
                layers.Dense(obs_dim, activation="tanh",
                             kernel_initializer=keras.initializers.glorot_uniform()),
                layers.Dense(32, None,
                             kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3)),
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


class Model(object):
    def __init__(self, obs_dim, act_dim, lr, gamma):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma

        self.actor = ActorNetwork(obs_dim=obs_dim, act_dim=act_dim, lr=self.lr)
        self.memory = Memory()

    def step(self, obs):
        if obs.ndim < 2:
            obs = obs[np.newaxis, :]
        prob_weights = self.actor.choose_action(obs).numpy()
        return np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

    def learn(self):
        obs = np.vstack(self.memory.ep_obs)
        act = np.array(self.memory.ep_act)
        rwd = np.array(self.memory.ep_rwd)

        discounted_rwd = self.discount_and_norm_rewards(rwd)

        with tf.GradientTape() as tape:
            cross_entropy = self.actor.get_cross_entropy(obs, act)
            loss = tf.reduce_mean(cross_entropy * discounted_rwd)
            grad = tape.gradient(loss, self.actor.model.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(grad, self.actor.model.trainable_variables))
        self.memory.reset()

    def discount_and_norm_rewards(self, rwd):
        R, ret_rwd = 0, np.zeros_like(rwd)
        for t in reversed(range(len(rwd))):
            R = R * self.gamma + rwd[t]
            ret_rwd[t] = R
        ret_rwd = np.array(ret_rwd)
        return (ret_rwd - np.mean(ret_rwd)) / np.std(ret_rwd)
