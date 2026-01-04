# Train DDPG agent for parallel parking with visual rendering
import sys
import os
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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


def visual_training_demo(resume=False, use_her=True):
    """
    Visual training demo that saves real training data to a CSV file for plotting.
    Includes logic to append to existing logs if resuming.
    """
    import pandas as pd 

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
    render_every_n_steps = 0     
    render_delay = 0.0             
    print_every = 10  
    
    if use_her:
        log_filename = "training_log.csv"
        mode_title = "WITH HER"
    else:
        log_filename = "training_log_no_her.csv"
        mode_title = "NO HER (BASELINE)"            
        
    print("=" * 80)
    print(f"VISUAL DDPG TRAINING - {mode_title}")
    print("=" * 80)
    print(f"Saving logs to: {log_filename}")

    warmup_steps = 5000           
    if resume:
        warmup_steps = 0
        episodes_to_run = 500
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[1/3] Initialising parallel parking environment...")
    parking_env = ParallelParkingEnv(config={}, render_mode=None)

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
    trainer = Trainer(parking_env, agent, use_her=use_her)
    
    # --- SETUP LOGGING ---
    training_log = [] 
    log_path = os.path.join(RESULTS_DIR, log_filename)
    
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
            agent.noise_scale = 0.05 
            for param_group in agent.actor_optimiser.param_groups:
                param_group['lr'] = 1e-5
            for param_group in agent.critic_optimiser.param_groups:
                param_group['lr'] = 2e-5
                
            trainer.best_success_rate = 60.0
            print("Learning rates adjusted for fine-tuning")

            # --- NEW: Load Previous CSV Data ---
            if os.path.exists(log_path):
                try:
                    df_existing = pd.read_csv(log_path)
                    training_log = df_existing.to_dict('records')
                    print(f"Resumed log: Loaded {len(training_log)} previous episodes.")
                except Exception as e:
                    print(f"Warning: Could not load previous log: {e}")
        else:
            print(f"{load_path} not found. Starting from scratch!")
   
    print(f"  Trainer ready")
    print(f"    - Rendering every {render_every_n_steps} steps")
    print(f"    - Warmup phase: {warmup_steps} steps")
    print(f"    - Logging data to: {log_path}")

    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80 + "\n")

    total_steps = 0

    try:
        # Loop starts from the end of the loaded log
        start_episode = len(training_log)
        
        for episode in range(start_episode, start_episode + episodes_to_run):
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

                if render_every_n_steps > 0 and episode_steps % render_every_n_steps == 0:
                    parking_env.render()
                    if render_delay > 0:
                        time.sleep(render_delay)

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

            # --- Save Data to Log List ---
            training_log.append({
                "Episode": episode,
                "Reward": episode_reward,
                "Success": 1 if is_success else 0,
                "Crash": 1 if is_crash else 0,
                "Steps": episode_steps
            })

            # --- Save to CSV every 10 episodes ---
            if episode % 10 == 0:
                df = pd.DataFrame(training_log)
                df.to_csv(log_path, index=False)

            trainer.print_episode_summary(episode, print_every)

        print("\n" + "=" * 80)
        print("VISUAL TRAINING FINISHED!")
        
        # --- Final Save ---
        df = pd.DataFrame(training_log)
        df.to_csv(log_path, index=False)
        print(f"Training data saved to: {log_path}")

        final_model_path = os.path.join(RESULTS_DIR, "ddpg_agent_final.pth")
        print(f"Final model saved: {final_model_path}")
        print(f"Best performance: Success {trainer.best_success_rate:.1f}% | Crash {trainer.best_crash_rate:.1f}%")
        print("=" * 80)
        agent.save(final_model_path)

    except KeyboardInterrupt:
        print("\n\nTraining stopped by user")
        print("Saving checkpoint and data log...")
        
        # --- Save data on interrupt ---
        df = pd.DataFrame(training_log)
        df.to_csv(log_path, index=False)
        print(f"Log saved: {log_path}")

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
    
def plot_training_results():
    """
    Option 5: Plots BOTH 'training_log.csv' (HER) and 'training_log_no_her.csv' (Baseline).
    Overlaying them creates the comparison graph required for the report.
    
    """
    her_path = os.path.join(RESULTS_DIR, "training_log.csv")
    no_her_path = os.path.join(RESULTS_DIR, "training_log_no_her.csv")
    
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 1. Plot HER Data (Blue)
    if os.path.exists(her_path):
        print(f"Loading HER data from {her_path}...")
        df_her = pd.read_csv(her_path)
        # Smoothing window
        df_her['Success Rate'] = df_her['Success'].rolling(window=50, min_periods=1).mean() * 100
        plt.plot(df_her['Episode'], df_her['Success Rate'], label='DDPG + HER (Ours)', color='blue', linewidth=2)
        plt.fill_between(df_her['Episode'], df_her['Success Rate'], alpha=0.1, color='blue')
    else:
        print(f"[Info] HER log not found ({her_path}). Run Option 1 to generate.")

    # 2. Plot No-HER Data (Red)
    if os.path.exists(no_her_path):
        print(f"Loading Baseline data from {no_her_path}...")
        df_no = pd.read_csv(no_her_path)
        # Smoothing window
        df_no['Success Rate'] = df_no['Success'].rolling(window=50, min_periods=1).mean() * 100
        plt.plot(df_no['Episode'], df_no['Success Rate'], label='Standard DDPG (No HER)', color='red', linewidth=2, linestyle='--')
        plt.fill_between(df_no['Episode'], df_no['Success Rate'], alpha=0.1, color='red')
    else:
        print(f"[Info] Baseline log not found ({no_her_path}). Run Option 6 to generate.")

    plt.title("Learning Curve Comparison: Impact of HER", fontsize=16, pad=20)
    plt.ylabel("Success Rate (Avg over 50 eps) %", fontsize=14)
    plt.xlabel("Episode", fontsize=14)
    plt.ylim(-5, 105)
    plt.xlim(left=0)
    plt.legend(loc='upper left', fontsize=12)

    plot_path = os.path.join(RESULTS_DIR, "comparison_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison graph saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    print("Pick training mode:")
    print("1. Train DDPG + HER (Saves to training_log.csv)")
    print("2. Quick visual test (1 episode)")
    print("3. Test saved model")
    print("4. Resume DDPG + HER")
    print("5. Plot Comparison Graph")
    print("6. Train Baseline No HER (Saves to training_log_no_her.csv)")  # <--- NEW

    choice = input("Enter choice (1-6): ").strip()

    if choice == "1":
        visual_training_demo(resume=False, use_her=True)
    elif choice == "2":
        quick_visual_test()
    elif choice == "3":
        model_input = input("Enter model path: ").strip()
        if not model_input.endswith(".pth"): model_input += ".pth"
        test_model(os.path.join(RESULTS_DIR, model_input))
    elif choice == "4":
        visual_training_demo(resume=True, use_her=True)
    elif choice == "5":
        plot_training_results()
    elif choice == "6":
        visual_training_demo(resume=False, use_her=False)
    else:
        print("Invalid choice.")