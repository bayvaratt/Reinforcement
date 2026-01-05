from typing import List, Union
import numpy as np
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddpg.ddpg_agent import DDPGAgent
from replays.replay_buffer import ReplayBuffer
from env import parallel_lot


class Trainer:
    """
    Handles environment interaction and training loop.Implements Hindsight Experience Replay (HER) logic.
    """

    def __init__(self, parking_env: parallel_lot.ParallelParkingEnv, agent: DDPGAgent, use_her: bool = True) -> None:
        self.parking_env = parking_env
        self.agent = agent
        # If use_her is True, k=4. If False, k=0, which disables HER)
        self.future_k = 4 if use_her else 0

        obs, _ = self.parking_env.reset()

        self.state_dim = obs.shape[0]
        self.action_dim = self.parking_env.action_space.shape[0]

        self.replay_buffer = ReplayBuffer(capacity=1000000, state_dim=self.state_dim, action_dim=self.action_dim)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_history = []
        self.crash_history = []

        self.best_success_rate = 0.0
        self.best_crash_rate = 100.0
        
        self.best_combined_score = -np.inf  
        self.best_success_models = {}
        self.best_safety_models = {}

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """
        Sparse reward function for HER:
        - +1.0 if goal is achieved (in parking spot and aligned)
        - -0.01 small step penalty to encourage efficiency
        """
        dx = abs(achieved_goal[0] - desired_goal[0])
        dy = abs(achieved_goal[1] - desired_goal[1])
        heading_diff = achieved_goal[2] - desired_goal[2]
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi

        # Check if goal is achieved
        is_in_box = (dx <= 3.0) and (dy <= 0.75)
        is_aligned = abs(heading_diff) < 0.4 
        
        if is_in_box and is_aligned:
            return 1.0  # Sparse positive reward for success
        else:
            return -0.01  # Small step penalty to encourage efficiency

    def replace_goal_in_state(self, state: np.ndarray, new_goal: np.ndarray, achieved_goal: np.ndarray) -> np.ndarray:
        """
        Replace goal positions and relative positions for HER.
        """
        modified_state = state.copy()
        
        # Update goal absolute coordinates
        modified_state[8] = new_goal[0] / 50.0
        modified_state[9] = new_goal[1] / 50.0

        # Get car heading from state
        car_heading = state[5] * np.pi
        
        # Update relative position (transform to car frame)
        car_x, car_y = achieved_goal[0], achieved_goal[1]
        
        rel_x_world = new_goal[0] - car_x
        rel_y_world = new_goal[1] - car_y
        
        cos_t = np.cos(car_heading)
        sin_t = np.sin(car_heading)
        
        modified_state[0] = (rel_x_world * cos_t + rel_y_world * sin_t) / 50.0
        modified_state[1] = (-rel_x_world * sin_t + rel_y_world * cos_t) / 50.0
        
        new_rel_theta = new_goal[2] - car_heading
        
        # Normalise to [-pi, pi]
        new_rel_theta = (new_rel_theta + np.pi) % (2 * np.pi) - np.pi
        
        modified_state[2] = new_rel_theta / np.pi

        return modified_state

    def get_achieved_goal(self, state: np.ndarray) -> np.ndarray:
        """
        Get Agent's current position (X, Y, Theta).
        """
        x = state[3] * 50.0
        y = state[4] * 50.0
        theta = state[5] * np.pi
        return np.array([x, y, theta], dtype=np.float32)
  
    def get_desired_goal(self, state: np.ndarray) -> np.ndarray:
        """
        Get target parking spot (X, Y, Theta).
        """
        x = state[8] * 50.0
        y = state[9] * 50.0
        
        current_theta = state[5] * np.pi
        rel_theta = state[2] * np.pi
        theta = current_theta + rel_theta
        
        return np.array([x, y, theta], dtype=np.float32)

    def print_episode_summary(self, episode: int, print_every: int) -> None:
        #Prints episode metrics and saves models based on performance
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(self.episode_rewards[-print_every:])
            avg_length = np.mean(self.episode_lengths[-print_every:])
            
            success_rate = np.mean(self.success_history[-print_every:]) * 100
            crash_rate = np.mean(self.crash_history[-print_every:]) * 100
            
            combined_score = success_rate - crash_rate
            
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(project_root, "results/parallel_parking")
            os.makedirs(results_dir, exist_ok=True)
            
            saved_model = False
            save_reasons = []  #Why we're saving the model
            
            if combined_score > self.best_combined_score:
                self.best_combined_score = combined_score
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"OPTIMAL_success_{success_rate:.1f}_crash_{crash_rate:.1f}_{timestamp}.pth"
                save_path = os.path.join(results_dir, filename)
                self.agent.save(save_path)
                save_reasons.append(f"OPTIMAL (score: {combined_score:.1f})")
                saved_model = True
            
            if success_rate >= 70.0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"HIGH_SUCCESS_success_{success_rate:.1f}_crash_{crash_rate:.1f}_{timestamp}.pth"
                save_path = os.path.join(results_dir, filename)
                self.agent.save(save_path)
                save_reasons.append(f"HIGH_SUCCESS (>70%)")
                saved_model = True
            
            if crash_rate <= 20.0 and success_rate >= 50.0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"SAFE_success_{success_rate:.1f}_crash_{crash_rate:.1f}_{timestamp}.pth"
                save_path = os.path.join(results_dir, filename)
                self.agent.save(save_path)
                save_reasons.append(f"SAFE (crash<20%)")
                saved_model = True
            
            if success_rate >= 60.0 and crash_rate <= 30.0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"MILESTONE_success_{success_rate:.1f}_crash_{crash_rate:.1f}_{timestamp}.pth"
                save_path = os.path.join(results_dir, filename)
                self.agent.save(save_path)
                save_reasons.append(f"MILESTONE (60%+ success, 30%- crash)")
                saved_model = True
            
            if saved_model:
                print(f"Model saved [{', '.join(save_reasons)}]")  # Notify user of save
            
            print(f"Episode {episode + 1:4d} | Reward: {avg_reward:7.2f} | Success Rate: {success_rate:5.1f}% | Crash Rate: {crash_rate:5.1f}% | Score: {combined_score:6.1f} | Steps: {avg_length:5.1f} | Noise: {self.agent.noise_scale:.3f}")

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
            episode_cache = []

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
            is_crash = info.get('is_crash', False) or info.get('is_out_of_bounds', False)  # Check crash or bounds

            self.success_history.append(is_success)
            self.crash_history.append(is_crash)

            self.print_episode_summary(episode, print_every)

    # Method to store episode with HER
    def store_episode(self, episode_cache: List[tuple]) -> None:
        cache_size = len(episode_cache)
        for idx, (state, action, reward, next_state, done, info) in enumerate(episode_cache):
            self.replay_buffer.add(state, action, reward, next_state, done)

            available_future = cache_size - idx - 1
            if available_future > 0:
                num_samples = min(self.future_k, available_future)
                future_indices = np.random.choice(range(idx + 1, cache_size), size=num_samples, replace=False)
                
                for future_idx in future_indices:
                    future_state, _, _, _, _, future_info = episode_cache[future_idx]
                    
                    if future_info.get('is_crash', False) or future_info.get('is_out_of_bounds', False):
                        continue  # Skip if future state is bad

                    new_goal = self.get_achieved_goal(future_state)
                    
                    achieved_goal_from_state = self.get_achieved_goal(state)
                    achieved_goal_from_next = self.get_achieved_goal(next_state)

                    her_state = self.replace_goal_in_state(state, new_goal, achieved_goal_from_state)
                    her_next_state = self.replace_goal_in_state(next_state, new_goal, achieved_goal_from_next)
                    
                    # Recalculate reward for the new goal
                    her_reward = self.compute_reward(achieved_goal_from_next, new_goal)

                    self.replay_buffer.add(her_state, action, her_reward, her_next_state, done)
