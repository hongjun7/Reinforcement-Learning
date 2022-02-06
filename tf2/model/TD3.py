import copy
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers


def to_tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)


class ReplayBuffer(object):
    def __init__(self, buffer_limit):
        self.buffer = []
        self.buffer_limit = buffer_limit
    
    def put(self, transition):
        if len(self.buffer) >= self.buffer_limit:
            self.buffer = self.buffer[self.buffer_limit // 5:]
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, min(len(self.buffer), n))
        states, actions, rewards, next_states, masks = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            
            # states and actions are lists (no-wrap)
            states.append(s)
            actions.append(a)
            next_states.append(s_prime)
            
            # rewards and masks are float values (wrap with list type)
            rewards.append([r])
            masks.append([done])
        
        states = to_tensor(states)
        actions = to_tensor(actions)
        rewards = to_tensor(rewards)
        next_states = to_tensor(next_states)
        masks = to_tensor(masks)
        return states, actions, rewards, next_states, masks
    
    def size(self):
        return len(self.buffer)


class Actor(tf.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(Actor, self).__init__()
        self.nn = keras.Sequential(
            [
                layers.Dense(state_dim, activation='relu'),
                layers.Dense(400, activation='relu'),
                layers.Dense(300, activation='relu'),
                layers.Dense(action_dim, activation='tanh')
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=lr)
    
    def forward(self, state):
        return self.nn(state)


class Critic(tf.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(Critic, self).__init__()
        self.nn = keras.Sequential(
            [
                layers.Dense(state_dim + action_dim, activation='relu'),
                layers.Dense(400, activation='relu'),
                layers.Dense(300, activation='relu'),
                layers.Dense(1, None)
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=lr)
    
    def forward(self, state, action):
        sa = tf.concat([state, action], -1)
        return self.nn(sa)


class Model(object):
    def __init__(self, state_dim, action_dim,
                 lr=0.001, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_period=2):
        self.actor = Actor(state_dim, action_dim, lr)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.q1 = Critic(state_dim, action_dim, lr)
        self.q1_target = copy.deepcopy(self.q1)
        
        self.q2 = Critic(state_dim, action_dim, lr)
        self.q2_target = copy.deepcopy(self.q2)
        
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_period = policy_period
        
        self.n = 0
    
    def step(self, state):
        state = to_tensor(state.reshape(1, -1))
        return self.actor.forward(state).numpy().flatten()
    
    def soft_update(self, net, net_target):
        w = []
        param_target = net_target.nn.weights
        for i, param in enumerate(net.nn.weights):
            w.append(param * self.tau + param_target[i] * (1.0 - self.tau))
        net_target.nn.set_weights(w)
    
    def train(self, memory, gradient_steps, batch_size):
        self.n += 1
        for epoch in range(gradient_steps):
            # sample replay buffer
            state, action, reward, next_state, done = memory.sample(batch_size)

            with tf.GradientTape() as q1_tape, tf.GradientTape() as q2_tape:
                # select action according to policy and add clipped noise
                noise = tf.random.normal(shape=action.shape) * self.policy_noise
                noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
    
                next_action = self.actor_target.forward(next_state) + noise
                next_action = tf.clip_by_value(next_action, -1, +1)
    
                # compute the target Q value
                target_Q1 = self.q1_target.forward(next_state, next_action)
                target_Q2 = self.q2_target.forward(next_state, next_action)
                target_Q = tf.minimum(target_Q1, target_Q2)
                target_Q = reward + tf.stop_gradient((1 - done) * self.gamma * target_Q)
                
                # get current Q estimates
                current_Q1 = self.q1.forward(state, action)
                current_Q2 = self.q2.forward(state, action)
                
                # compute critic loss
                q1_loss = keras.losses.MSE(current_Q1, target_Q)
                q2_loss = keras.losses.MSE(current_Q2, target_Q)
                
            # optimize the critic
            q1_grad = q1_tape.gradient(q1_loss, self.q1.nn.trainable_variables)
            q2_grad = q2_tape.gradient(q2_loss, self.q2.nn.trainable_variables)
            self.q1.optimizer.apply_gradients(zip(q1_grad, self.q1.nn.trainable_variables))
            self.q2.optimizer.apply_gradients(zip(q2_grad, self.q2.nn.trainable_variables))

            # delayed policy updates
            if epoch % self.policy_period == 0:
                # compute actor loss
                with tf.GradientTape() as actor_tape:
                    actor_loss = -tf.reduce_mean(self.q1.forward(state, self.actor.forward(state)))
    
                # optimize the actor
                actor_grad = actor_tape.gradient(actor_loss, self.actor.nn.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.nn.trainable_variables))
    
                # update the frozen target models
                self.soft_update(self.q1, self.q1_target)
                self.soft_update(self.q2, self.q2_target)
                self.soft_update(self.actor, self.actor_target)
