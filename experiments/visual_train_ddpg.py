import sys
import os
import numpy as np
import torch
import time

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seeding import set_seed

from ddpg.ddpg_agent import DDPGAgent
from ddpg.trainer import Trainer
from env.parallel_lot import ParallelParkingEnv


def visual_training_demo():
    """
    Visual training demonstration - watch the DDPG agent learn to parallel park!
    """

    print("=" * 80)
    print("VISUAL DDPG TRAINING - WATCH THE AGENT LEARN TO PARK!")
    print("=" * 80)

    print("\nGoal: Agent learns to parallel park between other cars")
    print("Controls: Close pygame window to stop training")
    print("Stats: Episode rewards and success rates displayed in terminal")
    print("Speed: Rendering every 10 steps")
    print("Warm-up: First 1000 steps are RANDOM actions to fill memory buffer\n")

    # --- Configuration for Visual Demo ---
    num_episodes = 1000            
    max_steps_per_episode = 200    
    batch_size = 256               
    render_every_n_steps = 10      
    render_delay = 0.0        
    print_every = 10
    
    warm_up_steps = 5000
    set_seed(42)
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[1/3] Initialising Parallel Parking Environment...")
    parking_env = ParallelParkingEnv(config={}, render_mode="human")

    # Get state and action dimensions
    obs, info = parking_env.reset()
    state_dim = obs.shape[0]
    action_dim = parking_env.action_space.shape[0]
    
    max_action = np.array([np.pi / 4, 0.5])

    print("  Environment ready with visual rendering")
    print(f"    - Goal slot: {info.get('goal_slot', 'N/A')} (look for striped lines)")
    print(f"    - Parked cars: {info.get('num_parked_cars', 'N/A')}")
    print(f"    - Max action per dimension: {max_action}")

    # --- Initialise DDPG Agent ---
    print("\n[2/3] Initialising DDPG Agent...")
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount_factor=0.99,
        soft_update_factor=0.005,
        device=device
    )
    print("  Agent initialised with exploration noise")
    print(f"    - Initial noise scale: {agent.noise_scale}")

    print("\n[3/3] Starting Visual Training...")
    trainer = Trainer(parking_env, agent)
    
    print(f"  Trainer ready")
    print(f"    - Rendering every {render_every_n_steps} steps")
    print(f"    - Warm-up phase: {warm_up_steps} steps")

    print("\n" + "=" * 80)
    print("TRAINING STARTED - WATCH THE AGENT LEARN!")
    print("=" * 80 + "\n")

    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    success_history = []
    crash_history = []

    try:
        for episode in range(num_episodes):
            obs, _ = parking_env.reset()
            state = obs
            
            agent.noise.reset()
            episode_reward = 0
            episode_steps = 0
            episode_cache = []  # To store transitions for HER

            for _ in range(max_steps_per_episode):
                
                # Warm-up logic...
                if total_steps < warm_up_steps:
                    action = parking_env.action_space.sample()
                else:
                    action = agent.select_action(state, noise=True)

                next_obs, reward, terminated, truncated, info = parking_env.step(action)
                next_state = next_obs
                done = terminated or truncated

                episode_cache.append((state, action, reward, next_state, done, info))

                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # Render
                if render_every_n_steps > 0 and episode_steps % render_every_n_steps == 0:
                    parking_env.render()
                    if render_delay > 0:
                        time.sleep(render_delay)

                if done:
                    break
            
            # --- End of Episode Updates ---
            
            trainer.store_episode(episode_cache)

            # 2. Train the agent (reduced frequency for stability)
            if total_steps >= warm_up_steps and len(trainer.replay_buffer) > batch_size:
                train_steps = min(episode_steps, 10)  # Cap at 10 training steps per episode
                for _ in range(train_steps):
                    agent.train(trainer.replay_buffer, batch_size)

            # 3. Decay noise
            agent.decay_noise()

            # 4. Logging
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            trainer.episode_rewards = episode_rewards

            # Track success and crash rates
            is_success = info.get('is_success', False)
            is_crash = info.get('is_crash', False) or info.get('is_out_of_bounds', False)
            success_history.append(is_success)
            crash_history.append(is_crash)

            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(episode_rewards[-print_every:])
                avg_length = np.mean(episode_lengths[-print_every:])
                success_rate = np.mean(success_history[-print_every:]) * 100
                crash_rate = np.mean(crash_history[-print_every:]) * 100
                phase = "WARM-UP" if total_steps < warm_up_steps else "TRAINING"
                
                print(
                    f"Episode {episode + 1:4d}/{num_episodes} [{phase}] | "
                    f"Reward: {avg_reward:7.2f} | "
                    f"Success: {success_rate:3.0f}% | "
                    f"Crash: {crash_rate:3.0f}% | "
                    f"Steps: {avg_length:5.1f} | "
                    f"Noise: {agent.noise_scale:.3f}"
                )

        print("\n" + "=" * 80)
        print("VISUAL TRAINING COMPLETED!")
        print(f"Final model saved as: ddpg_agent_final.pth")
        print(f"Best model saved by trainer (Success: {trainer.best_success_rate:.1f}% | Crash: {trainer.best_crash_rate:.1f}%)")
        print("=" * 80)
        agent.save("ddpg_agent_final.pth")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        checkpoint_path = "ddpg_agent_visual_checkpoint.pth"
        agent.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up...")
        parking_env.close()
        print("Environment closed")


def quick_visual_test():
    """
    Quick visual test - just watch one episode to see the environment
    """
    print("\nQUICK VISUAL TEST - One Episode Demo")
    print("=" * 50)
    
    set_seed(42)  # Ensure reproducible test results

    env = ParallelParkingEnv(render_mode="human")
    obs, info = env.reset()

    print(f"Goal slot: {info['goal_slot']} (striped lines)")
    print("Watch the car spawn and try to park...")
    print("Close pygame window when done\n")

    # Create a simple agent for demo
    agent = DDPGAgent(
        state_dim=obs.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=np.array([np.pi / 4, 1.0]),
    )

    try:
        for _ in range(100):  # 100 steps max
            action = agent.select_action(obs, noise=True)
            obs, reward, terminated, truncated, info = env.step(action)

            env.render()
            time.sleep(0.1)

            if terminated or truncated:
                status = "SUCCESS!" if info['is_success'] else "FAILED"
                print(f"Episode ended: {status} | Reward: {reward}")
                break

    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        env.close()
        print("Test completed!")


if __name__ == "__main__":
    print("Choose training mode:")
    print(f"1. Full visual training (1000 episodes)")
    print("2. Quick visual test (1 episode)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        quick_visual_test()
    else:
        visual_training_demo()