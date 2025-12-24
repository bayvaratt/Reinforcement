from typing import Union
from ddpg.ddpg_agent import DDPGAgent
from replays.replay_buffer import ReplayBuffer
from env import parallel_lot, perpindicular_lot

class Trainer:
    """
    Handles environment interaction and training loop.
    """
    def __init__(self, parking_env: Union[parallel_lot.ParallelParkingEnv, perpindicular_lot.CustomParkingEnv], agent: DDPGAgent, replay_buffer: ReplayBuffer) -> None:
        self.parking_env = parking_env
        self.agent = agent
        self.replay_buffer = replay_buffer

    def train(self, num_episodes: int, max_steps_per_episode: int, batch_size: int) -> None:
        """
        Run the training loop.
        """
        for episode in range(num_episodes):
            state, _ = self.parking_env.reset() # Might need changing after looking at env code
            self.agent.noise.reset()
            episode_reward = 0

            for _ in range(max_steps_per_episode):
                action = self.agent.select_action(state, noise=True)
                next_state, reward, terminated, truncated, _ = self.parking_env.step(action) # Might need changing after looking at env code
                self.replay_buffer.add(state, action, reward, next_state, terminated)
                state = next_state
                episode_reward += reward

                if len(self.replay_buffer) > batch_size: # Ensure enough samples before training
                    self.agent.train(self.replay_buffer, batch_size)

                if terminated or truncated:
                    break
            
            print(f"Episode {episode + 1}, Reward: {episode_reward}")
