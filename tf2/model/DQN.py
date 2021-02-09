import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, initializers
import random


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.last_index = 0
    
    def store_transition(self, obs_prv, act, rwd, obs_cur, done):
        data = (obs_prv, act, rwd, obs_cur, done)
        if self.last_index >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.last_index] = data
        self.last_index = (self.last_index + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_prv, act, rwd, obs_cur, done = map(np.stack, zip(*batch))
        return obs_prv, act, rwd, obs_cur, done


class ValueNetwork(object):
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.model = keras.Sequential(
            [
                layers.Dense(obs_dim),
                layers.Dense(10, activation="tanh", kernel_initializer=initializers.glorot_uniform()),
                layers.Dense(act_dim)
            ]
        )
    
    def step(self, obs):
        return self.model(obs)


class Model(object):
    def __init__(self, obs_dim, act_dim, lr, gamma, epsilon, tau=0.05, buffer_capacity=int(1e6)):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        
        self.Q = ValueNetwork(obs_dim=obs_dim, act_dim=act_dim)
        self.target_Q = ValueNetwork(obs_dim=obs_dim, act_dim=act_dim)
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.optimizer = optimizers.Adam(learning_rate=lr)
    
    def step(self, obs):
        if obs.ndim < 2: obs = obs[np.newaxis, :]
        action = self.Q.step(obs)
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(action, axis=1)[0]
        return action
    
    def learn(self):
        obs_prv, act, rwd, obs_nxt, done = self.memory.sample(batch_size=128)
        with tf.GradientTape(persistent=True) as tape:
            q_value = self.Q.step(obs_prv)
            action_one_hot = tf.one_hot(act, self.act_dim, dtype=tf.float32)
            q_value_one_hot = tf.reduce_sum(tf.multiply(q_value, action_one_hot), axis=1)
            target_Q = rwd + (1.-np.float32(done)) * self.gamma * tf.reduce_max(self.target_Q.step(obs_nxt), axis=1)

            loss = tf.reduce_mean(tf.square(q_value_one_hot - target_Q))
            
            grad = tape.gradient(loss, self.Q.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.Q.model.trainable_variables))
            
            for t, e in zip(self.target_Q.model.trainable_variables, self.Q.model.trainable_variables):
                t.assign(t * (1-self.tau) + e * self.tau)
