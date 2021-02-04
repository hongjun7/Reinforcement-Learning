import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python.distributions import Categorical
from tensorflow.keras import layers, optimizers


class Memory:
    def __init__(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []
        self.ep_logprob = []

    def store(self, obs, act, rwd, logprob):
        self.ep_obs.append(obs)
        self.ep_act.append(act)
        self.ep_rwd.append(rwd)
        self.ep_logprob.append(logprob)

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []
        self.ep_logprob = []


class ActorCritic(tf.Module):
    def __init__(self, obs_dim, act_dim, lr):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor = keras.Sequential(
            [
                layers.Dense(obs_dim),
                layers.Dense(64, activation=layers.LeakyReLU(), kernel_initializer=keras.initializers.he_uniform()),
                layers.Dense(64, activation=layers.LeakyReLU(), kernel_initializer=keras.initializers.he_uniform()),
                layers.Dense(act_dim, activation=layers.LeakyReLU())
            ]
        )
        self.critic = keras.Sequential(
            [
                layers.Dense(obs_dim),
                layers.Dense(64, activation=layers.LeakyReLU(), kernel_initializer=keras.initializers.he_uniform()),
                layers.Dense(64, activation=layers.LeakyReLU(), kernel_initializer=keras.initializers.he_uniform()),
                layers.Dense(1)
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=lr)

    def step(self, obs):
        if obs.ndim < 2:
            obs = obs[np.newaxis, :]
        action_logits = self.actor(obs)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action.numpy()[0], dist.log_prob(action)

    def infer(self, obs, act):
        action_logits = self.actor(obs)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(act)
        dist_entropy = dist.entropy()
        q_value = self.critic(obs)
        return action_logprobs, tf.squeeze(q_value), dist_entropy


class Model(object):
    def __init__(self, obs_dim, act_dim, lr, gamma, clip_range, update_ep_epochs):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.clip_range = clip_range
        self.MseLoss = keras.losses.mean_squared_error
        self.update_ep_epochs = update_ep_epochs

        self.policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, lr=lr)
        self.memory = Memory()

    def step(self, obs):
        return self.policy.step(obs)

    def learn(self):
        old_obs = np.vstack(self.memory.ep_obs)
        old_act = np.array(self.memory.ep_act)
        old_logprob = np.array(self.memory.ep_logprob).ravel()
        
        rwd = []
        discounted_rwd = 0
        for reward in reversed(self.memory.ep_rwd):
            discounted_rwd = reward + (self.gamma*discounted_rwd)
            rwd.insert(0, discounted_rwd)
        rwd = tf.constant(rwd, dtype=tf.float32)
        rwd = (rwd - tf.reduce_mean(rwd)) / tf.math.reduce_std(rwd)

        for epoch in range(self.update_ep_epochs):
            with tf.GradientTape() as tape:
                logprobs, q_value, entropy = self.policy.infer(old_obs, old_act)
                ratio = tf.exp(logprobs - tf.stop_gradient(old_logprob))
                advantages = (rwd - tf.stop_gradient(q_value)).numpy()
                advantages = (advantages - np.mean(advantages)) / np.std(advantages)
                advantages = tf.constant(advantages, dtype=tf.float32)
                
                # surrogate losses
                L1 = ratio * advantages
                L2 = tf.clip_by_value(ratio, 1-self.clip_range, 1+self.clip_range) * advantages
                
                loss = -tf.minimum(L1, L2) + 0.5*self.MseLoss(q_value, rwd) - 0.001*entropy
                gradient = tape.gradient(tf.reduce_mean(loss), self.policy.trainable_variables)
                self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.memory.reset()
