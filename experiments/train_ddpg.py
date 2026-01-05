# Train a DDPG agent for parallel parking without rendering
import sys
import os
import numpy as np
import torch
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "parallel_parking")
os.makedirs(RESULTS_DIR, exist_ok=True)

from ddpg import trainer
from utils.seeding import set_seed

from ddpg.ddpg_agent import DDPGAgent
from ddpg.trainer import Trainer
from env.parallel_lot import ParallelParkingEnv


def train_ddpg(resume=False):
    """
    Train DDPG agent for parallel parking without visual rendering.
    Runs faster without rendering overhead.
    """

    print("=" * 80)
    print("DDPG TRAINING - PARALLEL PARKING")
    print("=" * 80)

    print("\nGoal: Train agent to park between other cars")
    print("Output: Episode rewards and success rates shown in terminal")
    print("Speed: No rendering")
    print("Warmup: First 5000 steps are random actions to fill buffer\n")

    episodes_to_train = 2000            
    max_episode_steps = 200    
    batch_size = 256               
    print_every = 10               

    warmup_steps = 5000           
    if resume:
        warmup_steps = 0
        episodes_to_train = 500
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[1/3] Initialising parallel parking environment...")
    parking_env = ParallelParkingEnv(config={}, render_mode=None)

    obs, info = parking_env.reset()
    state_dim = obs.shape[0]
    action_dim = parking_env.action_space.shape[0]
    
    max_action = np.array([np.pi / 4, 1.0])

    print("  Environment ready (headless mode)")
    print(f"    - State dimension: {state_dim}")
    print(f"    - Action dimension: {action_dim}")
    print(f"    - Max action bounds: {max_action}")

    print("\n[2/3] Setting up DDPG agent...")
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount_factor=0.99,
        soft_update_factor=0.001, 
        device=device
    )
    print("  Agent ready with exploration noise")
    print(f"    - Starting noise scale: {agent.noise_scale}")

    print("\n[3/3] Beginning training...")
    trainer = Trainer(parking_env, agent)
    
    if resume:
        model_input = input("Enter the model file name or path to resume from (e.g., ddpg_agent_final.pth): ").strip()
        
        if os.path.isabs(model_input):
            load_path = model_input
        else:
            if model_input.startswith("results/parallel_parking/"):
                load_path = os.path.join(PROJECT_ROOT, model_input)
            else:
                load_path = os.path.join(RESULTS_DIR, model_input)
        
        if not load_path.endswith(".pth"):
            load_path = f"{load_path}.pth"
        
        if os.path.exists(load_path):
            print(f"Resuming training: Loading model from {load_path}")
            agent.load(load_path)
            agent.noise_scale = 0.1 
            for param_group in agent.actor_optimiser.param_groups:
                param_group['lr'] = 5e-6  # Lower actor learning rate
            for param_group in agent.critic_optimiser.param_groups:
                param_group['lr'] = 5e-5  # Lower critic learning rate

            trainer.replay_buffer.pointer = 0
            trainer.replay_buffer.size = 0
            print("Learning rates adjusted for fine-tuning")
            
        else:
            print(f"{load_path} not found. Starting from scratch!")
   
    print(f"  Trainer ready")
    print(f"    - Warmup phase: {warmup_steps} steps")

    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80 + "\n")

    total_steps = 0
    start_time = time.time()

    try:
        for episode in range(episodes_to_train):
            obs, _ = parking_env.reset()
            state = obs
            
            agent.noise.reset()
            episode_reward = 0
            episode_steps = 0
            episode_cache = []

            for _ in range(max_episode_steps):
                
                if total_steps < warmup_steps:
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

                if done:
                    break
            
            trainer.store_episode(episode_cache)

            if total_steps >= warmup_steps and len(trainer.replay_buffer) > batch_size:
                for _ in range(episode_steps):
                    agent.train(trainer.replay_buffer, batch_size)

            agent.noise_scale = max(0.02, agent.noise_scale * 0.998)

            trainer.episode_rewards.append(episode_reward)
            trainer.episode_lengths.append(episode_steps)

            is_success = info.get('is_success', False)
            is_crash = info.get('is_crash', False) or info.get('is_out_of_bounds', False)
            
            trainer.success_history.append(is_success)
            trainer.crash_history.append(is_crash)

            trainer.print_episode_summary(episode, print_every)

        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("TRAINING FINISHED!")
        final_model_path = os.path.join(RESULTS_DIR, "ddpg_agent_final.pth")
        print(f"Final model saved: {final_model_path}")
        print(f"Best performance: Success {trainer.best_success_rate:.1f}% | Crash {trainer.best_crash_rate:.1f}%")
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print("=" * 80)
        agent.save(final_model_path)

    except KeyboardInterrupt:
        print("\n\nTraining stopped by user")
        print("Saving checkpoint...")
        checkpoint_path = os.path.join(RESULTS_DIR, "ddpg_agent_checkpoint.pth")
        agent.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        parking_env.close()


def test_model(model_path: str):
    """Test a saved model for 100 episodes without rendering."""
    print(f"Testing model: {model_path}")
    successes = 0
    crashes = 0
    
    if not os.path.exists(model_path):
        print(f"Error: Can't find '{model_path}'.")
        return

    env = ParallelParkingEnv(config={}, render_mode=None)
    obs, info = env.reset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = np.array([np.pi / 4, 1.0])

    agent = DDPGAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        max_action=max_action,
        device=device
    )

    agent.load(model_path)
    print("Model loaded OK.")

    test_episodes = 100
    total_reward = 0
    
    for i in range(test_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        while not done:
            # Note: use noise=False for testing
            action = agent.select_action(obs, noise=False)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
        
        # Check result
        status = "SUCCESS" if info.get('is_success') else "FAIL"
        if info.get('is_crash') or info.get('is_out_of_bounds'): 
            status = "CRASH"
        
        if status == "SUCCESS":
            successes += 1
        elif status == "CRASH":
            crashes += 1
        
        total_reward += episode_reward
        
        if (i + 1) % 10 == 0:
            print(f"Episode {i+1}/{test_episodes} | {status} | Reward: {episode_reward:.2f}")

    success_rate = (successes / test_episodes) * 100
    crash_rate = (crashes / test_episodes) * 100
    avg_reward = total_reward / test_episodes
    
    print("\n" + "=" * 80)
    print(f"TEST RESULTS")
    print(f"Success: {success_rate:.1f}% | Crashes: {crash_rate:.1f}%")
    print(f"Avg reward: {avg_reward:.2f}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    print("Pick training mode:")
    print("1. Full training (2000 episodes, no rendering)")
    print("2. Test saved model (100 episodes)")
    print("3. Resume training (500 episodes)")

    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        train_ddpg(resume=False)
    elif choice == "2":
        model_input = input("Enter the model file name or path (e.g., ddpg_agent_final.pth): ").strip()
        
        if os.path.isabs(model_input):
            model_path = model_input
        else:
            if model_input.startswith("results/parallel_parking/"):
                model_path = os.path.join(PROJECT_ROOT, model_input)
            else:
                model_path = os.path.join(RESULTS_DIR, model_input)
        
        if not model_path.endswith(".pth"):
            model_path = f"{model_path}.pth"
        
        test_model(model_path)
    elif choice == "3":
        train_ddpg(resume=True)
    else:
        print("Invalid choice")