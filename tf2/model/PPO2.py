import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python.distributions import MultivariateNormalTriL
from tensorflow.keras import layers, optimizers, initializers


class Memory:
    def __init__(self):
        self.ep_obs, self.ep_act, self.ep_rwd, self.ep_done = [], [], [], []
        self.ep_logprob = []

    def store(self, obs, act, rwd, done, logprob):
        self.ep_obs.append(obs)
        self.ep_act.append(act)
        self.ep_rwd.append(rwd)
        self.ep_done.append(done)
        self.ep_logprob.append(logprob)

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd, self.ep_done = [], [], [], []
        self.ep_logprob = []


class ActorCritic(tf.Module):
    def __init__(self, obs_dim, act_dim, lr):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.std = 0.5
        self.action_var = tf.constant([self.std * self.std for i in range(act_dim)])
        self.cov_mat = tf.linalg.tensor_diag(self.action_var)
        actor_input = keras.Input(shape=(obs_dim,))
        actor_layer = layers.Dense(64, activation="tanh", kernel_initializer=initializers.he_uniform())(actor_input)
        actor_layer = layers.Dense(64, activation="tanh", kernel_initializer=initializers.he_uniform())(actor_layer)
        actor_output = layers.Dense(act_dim, activation="tanh")(actor_layer)
        self.actor = keras.Model(inputs=actor_input, outputs=actor_output)

        critic_input = keras.Input(shape=(obs_dim,))
        critic_layer = layers.Dense(64, activation="tanh", kernel_initializer=initializers.he_uniform())(critic_input)
        critic_layer = layers.Dense(64, activation="tanh", kernel_initializer=initializers.he_uniform())(critic_layer)
        critic_output = layers.Dense(1)(critic_layer)
        self.critic = keras.Model(inputs=critic_input, outputs=critic_output)
        
        self.optimizer = optimizers.Adam(learning_rate=lr)

    def step(self, obs):
        if obs.ndim < 2:
            obs = obs[np.newaxis, :]
        action_mean = self.actor(obs)
        dist = MultivariateNormalTriL(loc=action_mean, scale_tril=self.cov_mat)
        action = dist.sample()
        return action.numpy()[0], dist.log_prob(action)

    def infer(self, obs, act):
        if obs.ndim < 2:
            obs = obs[np.newaxis, :]
        action_mean = self.actor(obs)
        dist = MultivariateNormalTriL(loc=action_mean, scale_tril=self.cov_mat)
        action_logprobs = dist.log_prob(act)
        dist_entropy = dist.entropy()
        q_value = self.critic(obs)
        return action_logprobs, tf.squeeze(q_value), dist_entropy


class Model(object):
    def __init__(self, obs_dim, act_dim, lr, gamma, clip_range, update_ep_epochs, vf_coef=0.5, ent_coef=0.001):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.clip_range = clip_range
        self.MseLoss = keras.losses.mean_squared_error
        self.update_ep_epochs = update_ep_epochs
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, lr=lr)
        self.memory = Memory()

    def step(self, obs):
        return self.policy.step(obs)

    def learn(self):
        old_obs = np.vstack(self.memory.ep_obs)
        old_act = np.array(self.memory.ep_act)
        old_logprob = np.array(self.memory.ep_logprob).ravel()

        ep_reward = np.array(self.memory.ep_rwd)
        ep_done = np.array(self.memory.ep_done)
        
        rwd = []
        discounted_rwd = 0
        for reward, done in zip(reversed(ep_reward), reversed(ep_done)):
            if done:
                discounted_rwd = 0
            discounted_rwd = reward + (self.gamma*discounted_rwd)
            rwd.insert(0, discounted_rwd)
        rwd = tf.constant(rwd, dtype=tf.float32)

        for epoch in range(self.update_ep_epochs):
            with tf.GradientTape() as tape:
                logprobs, q_value, entropy = self.policy.infer(old_obs, old_act)
                ratio = tf.exp(logprobs - old_logprob)
                advs = rwd - q_value
                advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-6)
                
                # surrogate losses
                L1 = ratio * (-advs)
                L2 = tf.clip_by_value(ratio, 1-self.clip_range, 1+self.clip_range) * (-advs)
                
                loss_CLIP = tf.maximum(L1, L2)
                loss_VF = self.vf_coef * self.MseLoss(q_value, rwd)
                loss_ENT = self.ent_coef * (-entropy)
                
                loss = loss_CLIP + loss_VF + loss_ENT
                gradient = tape.gradient(loss, self.policy.trainable_variables)
                gradient, _ = tf.clip_by_global_norm(gradient, 0.5)
                self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.memory.reset()

    def save(self):
        self.policy.actor.save_weights('./weights/ppo-BipedalWalker/Actor')
        self.policy.critic.save_weights('./weights/ppo-BipedalWalker/Critic')
    
    def load(self):
        self.policy.actor.load_weights('./weights/ppo-BipedalWalker/Actor')
        self.policy.critic.load_weights('./weights/ppo-BipedalWalker/Critic')
