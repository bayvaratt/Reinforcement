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

    # --- VISUAL SETTINGS ---

    render_every_n_steps = 0   
    render_delay = 0.0          
    
    # Determine render mode based on settings
    mode_str = "human" if render_every_n_steps > 0 else None

    # 1. Setup Filenames
    if use_her:
        log_filename = "training_log.csv"
        model_filename = "ddpg_agent_final.pth"
        mode_title = "WITH HER (FAST MODE)"
    else:
        log_filename = "training_log_no_her.csv"
        model_filename = "ddpg_agent_no_her.pth"
        mode_title = "NO HER (BASELINE)"

    print("=" * 80)
    print(f"ACADEMIC DDPG TRAINING - {mode_title}")
    print(f"Rendering: {'ENABLED' if mode_str else 'DISABLED'} (Every {render_every_n_steps} steps)")
    print("=" * 80)
    print("Optimization: Reduced evaluation freq & gradient updates for speed.")
    print(f"Saving logs to: {log_filename}")

    episodes_to_run = 2000            
    max_episode_steps = 200    
    batch_size = 256

    # Evaluate every 50 episodes (No noise)
    eval_interval = 50      # Train for 50 eps
    eval_episodes = 10      # Test for 10 eps
    
    updates_per_step = 1 
    
    warmup_steps = 5000           
    if resume:
        warmup_steps = 0
        episodes_to_run = 500
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environments
    print("[1/3] Initialising environments...")
    # MODIFIED: Pass the determined render_mode here
    train_env = ParallelParkingEnv(config={}, render_mode=mode_str)
    eval_env = ParallelParkingEnv(config={}, render_mode=None) # Keep eval headless for speed

    # Agent
    print("[2/3] Setting up DDPG agent...")
    obs, info = train_env.reset()
    state_dim = obs.shape[0]
    action_dim = train_env.action_space.shape[0]
    max_action = np.array([np.pi / 4, 1.0])

    agent = DDPGAgent(state_dim, action_dim, max_action, device=device)
    trainer = Trainer(train_env, agent, use_her=use_her)

    # Logging & Resuming
    training_log = []
    log_path = os.path.join(RESULTS_DIR, log_filename)
    
    if resume and os.path.exists(log_path):
        try:
            df_existing = pd.read_csv(log_path)
            training_log = df_existing.to_dict('records')
            print(f"Resumed log: {len(training_log)} entries.")
        except Exception as e:
            print(f"Warning: Could not load log: {e}")

    if resume:
        load_path = os.path.join(RESULTS_DIR, model_filename)
        if os.path.exists(load_path):
            agent.load(load_path)
            agent.noise_scale = 0.05
            print(f"Weights loaded from {model_filename}")

    total_steps = 0
    start_cycle = len(training_log) 
    total_cycles = episodes_to_run // eval_interval
    global_episode = start_cycle * eval_interval 

    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80 + "\n")

    try:
        for cycle in range(start_cycle, total_cycles):
            
            # --- PHASE 1: TRAINING ---
            # Rolling stats for terminal output
            train_rewards = []
            train_successes = []
            train_crashes = []
            train_steps = []

            for _ in range(eval_interval):
                global_episode += 1
                
                obs, _ = train_env.reset()
                state = obs
                agent.noise.reset()
                episode_steps = 0
                episode_reward = 0
                episode_cache = []

                for _ in range(max_episode_steps):
                    if total_steps < warmup_steps:
                        action = train_env.action_space.sample()
                    else:
                        action = agent.select_action(state, noise=True)

                    next_obs, reward, terminated, truncated, info = train_env.step(action)
                    done = terminated or truncated
                    episode_cache.append((state, action, reward, next_obs, done, info))
                    
                    # --- NEW RENDERING LOGIC ---
                    if render_every_n_steps > 0 and total_steps % render_every_n_steps == 0:
                        train_env.render()
                        if render_delay > 0:
                            time.sleep(render_delay)
                    # ---------------------------

                    state = next_obs
                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1
                    if done: break
                
                trainer.store_episode(episode_cache)
                
                # Update stats lists
                is_success = info.get('is_success', False)
                is_crash = info.get('is_crash', False) or info.get('is_out_of_bounds', False)
                train_rewards.append(episode_reward)
                train_successes.append(1 if is_success else 0)
                train_crashes.append(1 if is_crash else 0)
                train_steps.append(episode_steps)

                # --- OPTIMIZED UPDATE LOOP ---
                if total_steps >= warmup_steps and len(trainer.replay_buffer) > batch_size:
                    n_updates = int(episode_steps * updates_per_step)
                    for _ in range(n_updates):
                        agent.train(trainer.replay_buffer, batch_size)
                
                agent.noise_scale = max(0.02, agent.noise_scale * 0.9995)
                
                # --- PRINT EVERY 10 EPISODES ---
                if global_episode % 10 == 0:
                    avg_reward = np.mean(train_rewards[-10:])
                    success_rate = np.mean(train_successes[-10:]) * 100
                    crash_rate = np.mean(train_crashes[-10:]) * 100
                    avg_len = np.mean(train_steps[-10:])
                    score = success_rate - crash_rate

                    print(f"Episode {global_episode:4d} | Reward: {avg_reward:7.2f} | Success Rate: {success_rate:5.1f}% | Crash Rate: {crash_rate:5.1f}% | Score: {score:6.1f} | Steps: {avg_len:5.1f} | Noise: {agent.noise_scale:.3f}")

            # --- PHASE 2: EVALUATION (Runs without rendering to save time) ---
            eval_stats = {'reward': 0, 'success': 0, 'crash': 0, 'steps': 0}
            
            for _ in range(eval_episodes):
                obs, _ = eval_env.reset()
                state = obs
                done = False
                ep_reward = 0
                ep_steps = 0
                while not done:
                    action = agent.select_action(state, noise=False)
                    state, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    ep_steps += 1
                
                eval_stats['reward'] += ep_reward
                eval_stats['steps'] += ep_steps
                if info.get('is_success', False): eval_stats['success'] += 1
                if info.get('is_crash', False) or info.get('is_out_of_bounds', False): eval_stats['crash'] += 1
            
            # Averages
            avg_reward = eval_stats['reward'] / eval_episodes
            success_rate = (eval_stats['success'] / eval_episodes) * 100
            crash_rate = (eval_stats['crash'] / eval_episodes) * 100
            avg_steps = eval_stats['steps'] / eval_episodes
            
            # --- SAVE LOGS ---
            training_log.append({
                "Episode": global_episode,
                "Reward": avg_reward,
                "Success": success_rate,
                "Crash": crash_rate,
                "Steps": avg_steps
            })
            
            df = pd.DataFrame(training_log)
            df = df[['Episode', 'Reward', 'Success', 'Crash', 'Steps']]
            df.to_csv(log_path, index=False)
            
            if cycle % 5 == 0:
                agent.save(os.path.join(RESULTS_DIR, model_filename))

        agent.save(os.path.join(RESULTS_DIR, model_filename))
        print("Training Finished.")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving data...")
        agent.save(os.path.join(RESULTS_DIR, model_filename.replace(".pth", "_checkpoint.pth")))
        pd.DataFrame(training_log).to_csv(log_path, index=False)
        print("Saved.")

    finally:
        train_env.close()
        eval_env.close()

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
    Plots learning curves from CSV logs.
    Auto-detects if data is 0-1 or 0-100 to prevent scaling errors.
    """
    her_path = os.path.join(RESULTS_DIR, "training_log.csv")
    no_her_path = os.path.join(RESULTS_DIR, "training_log_no_her.csv")
    
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Helper to clean and smooth data
    def process_data(df, col_name='Success', window=5):
        # 1. Force column to numeric (coerces errors/strings to NaN)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        # 2. Check if data is 0-1 (decimal) or 0-100 (percent)
        # If max value > 1.0, it's likely already a percentage.
        scale_factor = 1.0 if df[col_name].max() > 1.0 else 100.0
        
        # 3. Apply rolling mean and scaling
        # Note: If using Code A (Academic), data is sparse, so window=5 is good.
        # If using Code B (Every step), use window=50.
        return df[col_name].rolling(window=window, min_periods=1).mean() * scale_factor

    # 1. Plot HER Data (Blue)
    if os.path.exists(her_path):
        print(f"Loading HER data from {her_path}...")
        try:
            df_her = pd.read_csv(her_path)
            y_values = process_data(df_her, 'Success', window=5)
            
            plt.plot(df_her['Episode'], y_values, label='DDPG + HER (Ours)', color='blue', linewidth=2)
            plt.fill_between(df_her['Episode'], y_values, alpha=0.1, color='blue')
        except Exception as e:
            print(f"Error reading HER log: {e}")
    else:
        print(f"[Info] HER log not found ({her_path}).")

    # 2. Plot No-HER Data (Red)
    if os.path.exists(no_her_path):
        print(f"Loading Baseline data from {no_her_path}...")
        try:
            df_no = pd.read_csv(no_her_path)
            y_values = process_data(df_no, 'Success', window=5)
            
            plt.plot(df_no['Episode'], y_values, label='Standard DDPG (No HER)', color='red', linewidth=2, linestyle='--')
            plt.fill_between(df_no['Episode'], y_values, alpha=0.1, color='red')
        except Exception as e:
            print(f"Error reading Baseline log: {e}")
    else:
        print(f"[Info] Baseline log not found ({no_her_path}).")

    plt.title("Learning Curve Comparison: Impact of HER", fontsize=16, pad=20)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.xlabel("Episode", fontsize=14)
    
    # Force Y-axis to 0-100 range
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