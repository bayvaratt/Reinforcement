# Train DDPG agent for parallel parking with visual rendering
import sys
import os
import numpy as np
import torch
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up project root and results directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "parallel_parking")
os.makedirs(RESULTS_DIR, exist_ok=True)

from utils.seeding import set_seed

from ddpg.ddpg_agent import DDPGAgent
from ddpg.trainer import Trainer
from env.parallel_lot import ParallelParkingEnv


def visual_training_demo(resume=False):
    """
    Visual training demo for watching the DDPG agent learn to park.
    """

    print("=" * 80)
    print("VISUAL DDPG TRAINING - PARALLEL PARKING")
    print("=" * 80)

    print("\nGoal: Train agent to park between other cars")
    print("Display: Visual rendering shows agent movements")
    print("Feedback: Episode rewards and success rates in terminal")
    print("Speed: Renders every 10 steps for smooth viewing")
    print("Warmup: First 5000 steps are random to fill buffer\n")

    episodes_to_run = 2000            
    max_episode_steps = 200    
    batch_size = 256               
    render_every_n_steps = 10     
    render_delay = 0.0             
    print_every = 10               

    warmup_steps = 5000           
    if resume:
        warmup_steps = 0
        episodes_to_run = 500
    set_seed(42)
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[1/3] Initialising parallel parking environment...")
    parking_env = ParallelParkingEnv(config={}, render_mode="human")

    # Get state and action dimensions
    obs, info = parking_env.reset()
    state_dim = obs.shape[0]
    action_dim = parking_env.action_space.shape[0]
    
    max_action = np.array([np.pi / 4, 1.0])

    print("  Environment ready with visual rendering")
    print(f"    - Goal slot: {info.get('goal_slot', 'N/A')} (look for striped markings)")
    print(f"    - Parked cars: {info.get('num_parked_cars', 'N/A')}")
    print(f"    - Max action bounds: {max_action}")

    # Set up DDPG agent
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

    print("\n[3/3] Beginning visual training...")
    trainer = Trainer(parking_env, agent)
    
    if resume:
        model_input = input("Enter the model file name or path to resume from (e.g., ddpg_agent_final.pth): ").strip()
        
        if os.path.isabs(model_input):
            load_path = model_input
        else:
            # Check if user already included full path
            if model_input.startswith("results/parallel_parking/"):
                load_path = os.path.join(PROJECT_ROOT, model_input)
            else:
                load_path = os.path.join(RESULTS_DIR, model_input)
        
        if not load_path.endswith(".pth"):
            load_path = f"{load_path}.pth"
        
        if os.path.exists(load_path):
            print(f"Resuming training: Loading model from {load_path}")
            agent.load(load_path)
            agent.noise_scale = 0.05 
            for param_group in agent.actor_optimiser.param_groups:
                param_group['lr'] = 1e-5  # Lower step size for fine-tuning
            for param_group in agent.critic_optimiser.param_groups:
                param_group['lr'] = 2e-5  # Conservative learning rate
                
            trainer.best_success_rate = 60.0
            print("Learning rates adjusted for fine-tuning")
            
        else:
            print(f"{load_path} not found. Starting from scratch!")
   
    print(f"  Trainer ready")
    print(f"    - Rendering every {render_every_n_steps} steps")
    print(f"    - Warmup phase: {warmup_steps} steps")

    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80 + "\n")

    total_steps = 0

    try:
        for episode in range(episodes_to_run):
            obs, _ = parking_env.reset()
            state = obs
            
            agent.noise.reset()
            episode_reward = 0
            episode_steps = 0
            episode_cache = []  # Store transitions for HER

            for _ in range(max_episode_steps):
                
                # Warmup phase: random actions
                if total_steps < warmup_steps:
                    action = parking_env.action_space.sample()
                else:
                    action = agent.select_action(state, noise=True)  # Policy with exploration

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
            
            trainer.store_episode(episode_cache)

            if total_steps >= warmup_steps and len(trainer.replay_buffer) > batch_size:
                for _ in range(episode_steps):
                    agent.train(trainer.replay_buffer, batch_size)  # Gradient updates

            agent.noise_scale = max(0.02, agent.noise_scale * 0.998)

            # Store episode stats
            trainer.episode_rewards.append(episode_reward)
            trainer.episode_lengths.append(episode_steps)

            is_success = info.get('is_success', False)
            is_crash = info.get('is_crash', False) or info.get('is_out_of_bounds', False)
            
            trainer.success_history.append(is_success)
            trainer.crash_history.append(is_crash)  # Track failure modes

            trainer.print_episode_summary(episode, print_every)

        print("\n" + "=" * 80)
        print("VISUAL TRAINING FINISHED!")
        final_model_path = os.path.join(RESULTS_DIR, "ddpg_agent_final.pth")
        print(f"Final model saved: {final_model_path}")
        print(f"Best performance: Success {trainer.best_success_rate:.1f}% | Crash {trainer.best_crash_rate:.1f}%")  # Results
        print("=" * 80)
        agent.save(final_model_path)

    except KeyboardInterrupt:
        print("\n\nTraining stopped by user")
        print("Saving checkpoint...")
        checkpoint_path = os.path.join(RESULTS_DIR, "ddpg_agent_visual_checkpoint.pth")
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
    Quick visual test - one episode demo of the environment.
    """
    print("\nQUICK VISUAL TEST - One Episode Demo")
    print("=" * 50)
    
    set_seed(42)

    env = ParallelParkingEnv(render_mode="human")
    obs, info = env.reset()

    print(f"Goal slot: {info['goal_slot']} (striped markings)")
    print("Watch the car spawn and try to park...")
    print("Close the window when done\n")

    # Simple agent for demo
    agent = DDPGAgent(
        state_dim=obs.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=np.array([np.pi / 4, 1.0]),
    )

    try:
        for _ in range(100):
            action = agent.select_action(obs, noise=True)
            obs, reward, terminated, truncated, info = env.step(action)

            env.render()
            time.sleep(0.1)  # Slow down for viewing

            if terminated or truncated:
                status = "SUCCESS!" if info['is_success'] else "FAILED"
                print(f"Episode result: {status} | Reward: {reward}")
                break

    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        env.close()
        print("Test completed!")

def test_model(model_path: str):
    """Test a saved model for 100 episodes with visual rendering."""
    print(f"--- Testing model: {model_path} ---")
    successes = 0
    crashes = 0
    
    if not os.path.exists(model_path):
        print(f"Error: File '{model_path}' not found.")
        return

    env = ParallelParkingEnv(config={}, render_mode="human")
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

    # Load the model
    agent.load(model_path)
    print("Model loaded OK.")

    # Test runs
    test_episodes = 100
    
    for i in range(test_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        print(f"\nTest episode {i+1}...")
        
        while not done:
            # Note: noise=False for testing
            action = agent.select_action(obs, noise=False)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step += 1
            
            env.render()
        
        # Check result
        status = "SUCCESS" if info.get('is_success') else "FAIL"
        if info.get('is_crash') or info.get('is_out_of_bounds'): 
            status = "CRASH"  # Hit something or went out
        
        if status == "SUCCESS":
            successes += 1
        elif status == "CRASH":
            crashes += 1
            
        print(f"Result: {status} | Reward: {total_reward:.2f} | Steps: {step}")

    success_rate = (successes / test_episodes) * 100
    crash_rate = (crashes / test_episodes) * 100
    print("\nSuccess: {:.1f}% | Crashes: {:.1f}%".format(success_rate, crash_rate))  # Final stats

    env.close()

if __name__ == "__main__":
    print("Pick training mode:")
    print("1. Full visual training (2000 episodes, with rendering)")
    print("2. Quick visual test (1 episode)")
    print("3. Test saved model (100 episodes)")
    print("4. Resume visual training (500 episodes)")

    choice = input("Enter choice (1, 2, 3, or 4): ").strip()

    if choice == "1":
        visual_training_demo(resume=False)
    elif choice == "2":
        quick_visual_test()
    elif choice == "3":
        model_input = input("Enter the model file name or path (e.g., ddpg_agent_final.pth): ").strip()
        
        if os.path.isabs(model_input):
            model_path = model_input
        else:
            # Check if user already included full path
            if model_input.startswith("results/parallel_parking/"):
                model_path = os.path.join(PROJECT_ROOT, model_input)
            else:
                model_path = os.path.join(RESULTS_DIR, model_input)
        
        if not model_path.endswith(".pth"):
            model_path = f"{model_path}.pth"
        
        test_model(model_path)
    else:
        visual_training_demo(resume=True)