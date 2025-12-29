from typing import List, Union
import numpy as np
import os
import sys
import time

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddpg.ddpg_agent import DDPGAgent
from replays.replay_buffer import ReplayBuffer
from env import parallel_lot


class Trainer:
    """
    Handles environment interaction and training loop - handles HER logic.
    """

    def __init__(self, parking_env: parallel_lot.ParallelParkingEnv, agent: DDPGAgent) -> None:
        self.parking_env = parking_env
        self.agent = agent
        self.future_k = 4  # Number of future steps to sample for HER - Used in the replay buffer sampling and is an empirical choice

        obs, _ = self.parking_env.reset()

        self.state_dim = obs.shape[0]
        self.action_dim = self.parking_env.action_space.shape[0]

        self.replay_buffer = ReplayBuffer(capacity=1000000, state_dim=self.state_dim, action_dim=self.action_dim)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_history = []
        self.crash_history = []

        self.best_success_rate = 0.0

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        reward = -0.15 # Base penalty
        
        normalised_distance = min(distance / 50.0, 1.0)
        
        reward += (1.0 - normalised_distance) * 0.1 
        
        if distance < 3.0:
            reward += 50.0
            
        return reward

    def replace_goal_in_state(self, state: np.ndarray, new_goal: np.ndarray, achieved_goal: np.ndarray) -> np.ndarray:
        """
        Replace goal positions and relative positions for HER.
        new_goal and achieved_goal are now both 2D (x, y).
        """
        modified_state = state.copy()
        
        # 1. Update Goal Absolute Coordinates (Indices -2, -1)
        modified_state[-2] = new_goal[0] / 30.0
        modified_state[-1] = new_goal[1] / 20.0

        # 2. Get car heading from state (normalised value * pi)
        car_heading = state[5] * np.pi
        
        # 3. Update Relative Position (Indices 0, 1)
        car_x, car_y = achieved_goal[0], achieved_goal[1]
        
        rel_x_world = new_goal[0] - car_x
        rel_y_world = new_goal[1] - car_y
        
        cos_t = np.cos(car_heading)
        sin_t = np.sin(car_heading)
        
        # Rotate world difference into car frame
        modified_state[0] = (rel_x_world * cos_t + rel_y_world * sin_t) / 30.0
        modified_state[1] = (-rel_x_world * sin_t + rel_y_world * cos_t) / 30.0
        
        # Note: Relative heading (index 2) stays the same since we're only changing position goals

        return modified_state

    def get_achieved_goal(self, state: np.ndarray) -> np.ndarray:
        """
        Get Agent's current position (X, Y) - consistent 2D for HER.
        """
        x = state[3] * 30.0
        y = state[4] * 20.0
        return np.array([x, y], dtype=np.float32)
  
    def get_desired_goal(self, state: np.ndarray) -> np.ndarray:
       """
       Get target parking spot (X,Y) - consistent 2D for HER.
       Indices [-2] and [-1] are Goal X, Goal Y normalised by 30 and 20.
       """
       x = state[-2] * 30.0
       y = state[-1] * 20.0
       return np.array([x, y], dtype=np.float32)

    def print_episode_summary(self, episode: int, print_every: int) -> None:
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(self.episode_rewards[-print_every:])
            avg_length = np.mean(self.episode_lengths[-print_every:])
            
            # Convert to percentages for easier reading
            success_rate = np.mean(self.success_history[-print_every:]) * 100
            crash_rate = np.mean(self.crash_history[-print_every:]) * 100
            
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                save_path = f"best_ddpg_model_success_{self.best_success_rate:.2f}.pth"
                self.agent.save(save_path)
                print(f"New best model saved: {save_path}")

            print(
                f"Episode {episode + 1:4d} | "
                f"Reward: {avg_reward:7.2f} | "
                f"Success: {success_rate:3.0f}% | "
                f"Crash: {crash_rate:3.0f}% | "
                f"Steps: {avg_length:5.1f} | "
                f"Noise: {self.agent.noise_scale:.3f}"
            )

    def train(
        self,
        num_episodes: int,
        max_steps_per_episode: int,
        batch_size: int,
        print_every: int = 10,
        render_every_n_steps: int = 1,
        render_delay: float = 0.0,
    ) -> None:
        """
        Run the training loop.
        """
        for episode in range(num_episodes):
            obs, _ = self.parking_env.reset()

            state = obs

            self.agent.noise.reset()
            episode_reward = 0
            episode_steps = 0
            episode_cache = []  # To store transitions for HER

            for _ in range(max_steps_per_episode):
                action = self.agent.select_action(state, noise=True)
                next_obs, reward, terminated, truncated, info = self.parking_env.step(action)
                next_state = next_obs

                done = terminated or truncated

                episode_cache.append((state, action, reward, next_state, done, info))

                state = next_state
                episode_reward += reward
                episode_steps += 1

                if render_every_n_steps > 0 and episode_steps % render_every_n_steps == 0:
                    self.parking_env.render()
                    if render_delay > 0:
                        time.sleep(render_delay)

                if done or truncated:
                    break

            # Store episode transitions with HER after episode ends
            self.store_episode(episode_cache)

            # Train once per episode if buffer has enough samples
            if len(self.replay_buffer) > batch_size:
                self.agent.train(self.replay_buffer, batch_size)

            # Decay noise at the end of each episode
            self.agent.decay_noise()

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)

            is_success = info.get('is_success', False)
            is_crash = info.get('is_crash', False) or info.get('is_out_of_bounds', False)

            self.success_history.append(is_success)
            self.crash_history.append(is_crash)

            self.print_episode_summary(episode, print_every)

    # method to store episode with HER
    def store_episode(self, episode_cache: List[tuple]) -> None:
        cache_size = len(episode_cache)
        for idx, (state, action, reward, next_state, done, info) in enumerate(episode_cache):
            # 1. Store the REAL experience (with the -40 crash penalty)
            self.replay_buffer.add(state, action, reward, next_state, done)

            # 2. HER Logic (Hindsight Experience Replay)
            available_future = cache_size - idx - 1
            if available_future > 0:
                num_samples = min(self.future_k, available_future)
                future_indices = np.random.choice(range(idx + 1, cache_size), size=num_samples, replace=False)
                
                for future_idx in future_indices:
                    future_state, _, _, _, _, future_info = episode_cache[future_idx]
                    
                    if future_info.get('is_crash', False) or future_info.get('is_out_of_bounds', False):
                        continue

                    new_goal = self.get_achieved_goal(future_state)
                    
                    achieved_goal_from_state = self.get_achieved_goal(state)
                    achieved_goal_from_next = self.get_achieved_goal(next_state)

                    her_state = self.replace_goal_in_state(state, new_goal, achieved_goal_from_state)
                    her_next_state = self.replace_goal_in_state(next_state, new_goal, achieved_goal_from_next)
                    
                    # Recalculate reward for the new goal
                    her_reward = self.compute_reward(achieved_goal_from_next, new_goal)

                    self.replay_buffer.add(her_state, action, her_reward, her_next_state, done)
