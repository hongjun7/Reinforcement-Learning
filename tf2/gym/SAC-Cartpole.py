import gym
import logging
import numpy as np
import tensorflow as tf

from tf2.model.SAC import SoftActorCritic, ReplayBuffer

tf.keras.backend.set_floatx('float64')

logging.basicConfig(level='INFO')

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':
    seed = 42
    env_name = 'CartPole-v0'
    render = False
    verbose = False
    model_path = '../data/models/' + env_name + '/'
    batch_size = 128
    epochs = 50
    start_steps = 0
    gamma = 0.99
    tau = 0.995
    learning_rate = 0.001
    
    writer = tf.summary.create_file_writer(model_path + 'summary')
    
    # Instantiate the environment.
    env = gym.make(env_name)
    env.seed(seed)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    # Initialize Replay buffer.
    replay = ReplayBuffer(state_space, len([action_space]))
    
    # Initialize policy and Q-function parameters.
    sac = SoftActorCritic(state_space, action_space, writer,
                          learning_rate=learning_rate, gamma=gamma, tau=tau)
    
    # sac.policy.load_weights(args.model_path + '/2020-05-30-19:03:13.833421/model')
    
    # Repeat until convergence
    global_step = 1
    episode = 1
    episode_rewards = []
    while True:
        
        # Observe state
        current_state = env.reset()
        
        step = 1
        episode_reward = 0
        done = False
        while not done:
            
            if render:
                env.render()
            
            if global_step < start_steps:
                if np.random.uniform() > 0.8:
                    action = env.action_space.sample()
                else:
                    action = sac.sample_action(current_state)
            else:
                action = sac.sample_action(current_state)
            # Execute action, observe next state and reward
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if verbose:
                logging.info(f'Global step: {global_step}')
                logging.info(f'current_state: {current_state}')
                logging.info(f'action: {action}')
                logging.info(f'reward: {reward}')
                logging.info(f'next_state: {next_state}')
                logging.info(f'end: {done}')
            
            # Store transition in replay buffer
            replay.store(current_state, action, reward, next_state, done)
            
            # Update current state
            current_state = next_state
            
            step += 1
            global_step += 1
        
        if (step % 1 == 0) and (global_step > start_steps):
            for epoch in range(epochs):
                
                # Randomly sample minibatch of transitions from replay buffer
                current_states, actions, rewards, next_states, ends = replay.fetch_sample(num_samples=batch_size)
                
                # Perform single step of gradient descent on Q and policy network
                critic1_loss, critic2_loss, actor_loss, alpha_loss = sac.train(current_states, actions, rewards,
                                                                               next_states, ends)
                if verbose:
                    print(episode, global_step, epoch, critic1_loss.numpy(),
                          critic2_loss.numpy(), actor_loss.numpy(), episode_reward)
                
                with writer.as_default():
                    tf.summary.scalar("actor_loss", actor_loss, sac.epoch_step)
                    tf.summary.scalar("critic1_loss", critic1_loss, sac.epoch_step)
                    tf.summary.scalar("critic2_loss", critic2_loss, sac.epoch_step)
                    tf.summary.scalar("alpha_loss", alpha_loss, sac.epoch_step)
                
                sac.epoch_step += 1
                
                if sac.epoch_step % 1 == 0:
                    sac.update_weights()
        
        if episode % 1 == 0:
            sac.policy.save_weights(model_path + 'model')
        
        episode_rewards.append(episode_reward)
        episode += 1
        avg_episode_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])
        
        print(f"Episode {episode} reward: {episode_reward}")
        print(f"{episode} Average episode reward: {avg_episode_reward}")
        with writer.as_default():
            tf.summary.scalar("episode_reward", episode_reward, episode)
            tf.summary.scalar("avg_episode_reward", avg_episode_reward, episode)
