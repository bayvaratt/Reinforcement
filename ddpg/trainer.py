from typing import Union
import numpy as np
from ddpg.ddpg_agent import DDPGAgent
from replays.replay_buffer import ReplayBuffer
from env import parallel_lot, perpindicular_lot

class Trainer:
    """
    Handles environment interaction and training loop.
    """
    def __init__(self, parking_env: Union[parallel_lot.ParallelParkingEnv, perpindicular_lot.CustomParkingEnv], agent: DDPGAgent) -> None:
        self.parking_env = parking_env
        self.agent = agent
        
        obs, _ = self.parking_env.reset()
        
        self.replay_buffer = ReplayBuffer(capacity=1000000, state_dim=obs.flatten().shape[0], action_dim=self.parking_env.action_space.shape[0])
        self.episode_rewards = []
        self.episode_lengths = []

    def train(self, num_episodes: int, max_steps_per_episode: int, batch_size: int, print_every: int = 10) -> None:
        """
        Run the training loop.
        """
        for episode in range(num_episodes):
            obs, _ = self.parking_env.reset()
            state = obs.flatten() if len(obs.shape) > 1 else obs
            self.agent.noise.reset()
            episode_reward = 0
            episode_steps = 0

            for _ in range(max_steps_per_episode):
                action = self.agent.select_action(state, noise=True)
                
                next_obs, reward, terminated, truncated, _ = self.parking_env.step(action)
                next_state = next_obs.flatten() if len(next_obs.shape) > 1 else next_obs
                
                self.replay_buffer.add(state, action, reward, next_state, terminated)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1

                # Check if enough samples are available in replay buffer and train every 2 steps to prevent performance bottleneck
                if len(self.replay_buffer) > batch_size and episode_steps % 2 == 0:
                    self.agent.train(self.replay_buffer, batch_size)

                if terminated or truncated:
                    break
            
            # Decay noise at the end of each episode
            self.agent.decay_noise()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(self.episode_rewards[-print_every:])
                avg_length = np.mean(self.episode_lengths[-print_every:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward} | "
                      f"Avg Length: {avg_length} | "
                      f"Buffer Size: {len(self.replay_buffer)} | "
                      f"Noise Scale: {self.agent.noise_scale}")
                
                self.agent.save(f"ddpg_agent_episode_{episode + 1}.pth")