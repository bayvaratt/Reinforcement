# Train DDPG agent for parallel parking with visual rendering
import sys
import os
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "parallel_parking")
os.makedirs(RESULTS_DIR, exist_ok=True)

from utils.seeding import set_seed

from ddpg.ddpg_agent import DDPGAgent
from ddpg.trainer import Trainer
from env.parallel_lot import ParallelParkingEnv

def visual_training_demo(resume=False, use_her=True, rendering: bool = False):
    """
    Visual training demo that saves real training data to a CSV file for plotting.
    Includes logic to append to existing logs if resuming.
    """
    render_every_n_steps = 10 if rendering else 0   
    render_delay = 0.0  
    
    mode_str = "human" if render_every_n_steps > 0 else None

    if use_her:
        log_filename = "training_log.csv"
        model_filename = "ddpg_agent_final.pth"
        mode_title = "WITH HER (FAST MODE)"
    else:
        log_filename = "training_log_no_her.csv"
        model_filename = "ddpg_agent_no_her.pth"
        mode_title = "NO HER (BASELINE)"

    print("=" * 80)
    print(f"DDPG TRAINING - {mode_title}")
    print(f"Rendering: {'ENABLED' if mode_str else 'DISABLED'} (Every {render_every_n_steps} steps)")
    print("=" * 80)
    print(f"Saving logs to: {log_filename}")

    episodes_to_run = 2000            
    max_episode_steps = 200    
    batch_size = 256

    eval_interval = 50 
    eval_episodes = 10 
    
    updates_per_step = 1 
    
    warmup_steps = 5000           
    if resume:
        warmup_steps = 0
        episodes_to_run = 500
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("[1/3] Initialising environments...")
    train_env = ParallelParkingEnv(config={}, render_mode=mode_str)
    eval_env = ParallelParkingEnv(config={}, render_mode=None)

    print("[2/3] Setting up DDPG agent...")
    obs, info = train_env.reset()
    state_dim = obs.shape[0]
    action_dim = train_env.action_space.shape[0]
    max_action = np.array([np.pi / 4, 1.0])

    agent = DDPGAgent(state_dim, action_dim, max_action, device=device)
    trainer = Trainer(train_env, agent, use_her=use_her)

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
                    
                    if render_every_n_steps > 0 and total_steps % render_every_n_steps == 0:
                        train_env.render()
                        if render_delay > 0:
                            time.sleep(render_delay)

                    state = next_obs
                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1
                    if done: break
                
                trainer.store_episode(episode_cache)
                
                is_success = info.get('is_success', False)
                is_crash = info.get('is_crash', False) or info.get('is_out_of_bounds', False)
                train_rewards.append(episode_reward)
                train_successes.append(1 if is_success else 0)
                train_crashes.append(1 if is_crash else 0)
                train_steps.append(episode_steps)

                if total_steps >= warmup_steps and len(trainer.replay_buffer) > batch_size:
                    n_updates = int(episode_steps * updates_per_step)
                    for _ in range(n_updates):
                        agent.train(trainer.replay_buffer, batch_size)
                
                agent.noise_scale = max(0.02, agent.noise_scale * 0.9995)
                
                if global_episode % 10 == 0:
                    avg_reward = np.mean(train_rewards[-10:])
                    success_rate = np.mean(train_successes[-10:]) * 100
                    crash_rate = np.mean(train_crashes[-10:]) * 100
                    avg_len = np.mean(train_steps[-10:])
                    score = success_rate - crash_rate

                    print(f"Episode {global_episode:4d} | Reward: {avg_reward:7.2f} | Success Rate: {success_rate:5.1f}% | Crash Rate: {crash_rate:5.1f}% | Score: {score:6.1f} | Steps: {avg_len:5.1f} | Noise: {agent.noise_scale:.3f}")

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
            
            avg_reward = eval_stats['reward'] / eval_episodes
            success_rate = (eval_stats['success'] / eval_episodes) * 100
            crash_rate = (eval_stats['crash'] / eval_episodes) * 100
            avg_steps = eval_stats['steps'] / eval_episodes
            
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
            time.sleep(0.1)

            if terminated or truncated:
                status = "SUCCESS!" if info['is_success'] else "FAILED"
                print(f"Episode result: {status} | Reward: {reward}")
                break

    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        env.close()
        print("Test completed!")

def test_model(model_path: str, rendering: bool = False):
    """Test a saved model for 100 episodes with optional visual rendering."""
    print(f"--- Testing model: {model_path} ---")
    successes = 0
    crashes = 0
    
    if not os.path.exists(model_path):
        print(f"Error: File '{model_path}' not found.")
        return

    render_mode = "human" if rendering else None
    env = ParallelParkingEnv(config={}, render_mode=render_mode)
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
    
    for i in range(test_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        print(f"\nTest episode {i+1}...")
        
        while not done:
            action = agent.select_action(obs, noise=False)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step += 1
            
            if rendering:
                env.render()
        
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
    print("\nSuccess: {:.1f}% | Crashes: {:.1f}%".format(success_rate, crash_rate))

    env.close()
    
def plot_training_results():
    """
    Plots learning curves from CSV logs.Auto-detects if data is 0-1 or 0-100 to prevent scaling errors.
    """
    her_path = os.path.join(RESULTS_DIR, "training_log_final.csv")
    no_her_path = os.path.join(RESULTS_DIR, "training_log_no_her.csv")
    
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Helper to clean and smooth data
    def process_data(df, col_name='Success', window=5):
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        scale_factor = 1.0 if df[col_name].max() > 1.0 else 100.0
        
        return df[col_name].rolling(window=window, min_periods=1).mean() * scale_factor

    # 1. Plot HER Data (Blue)
    if os.path.exists(her_path):
        print(f"Loading HER data from {her_path}...")
        try:
            df_her = pd.read_csv(her_path)
            y_values = process_data(df_her, 'Success', window=5)
            
            plt.plot(df_her['Episode'], y_values, label='DDPG + HER', color='blue', linewidth=2)
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
    plt.ylim(-5, 105)
    plt.xlim(left=0)
    
    plt.legend(loc='upper left', fontsize=12)

    plot_path = os.path.join(RESULTS_DIR, "comparison_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison graph saved to: {plot_path}")
    plt.show()

def plot_detailed_comparisons():
    """
    Generates professional comparative graphs for the report:
    1. Average Reward: HER (Blue) vs Baseline (Red)
    2. Episode Length: HER (Blue) vs Baseline (Red)
    3. Behaviour Profile: HER Only (Stacked Area Chart)
    """
    her_path = os.path.join(RESULTS_DIR, "training_log_final.csv")
    no_her_path = os.path.join(RESULTS_DIR, "training_log_no_her.csv")
    
    def load_and_smooth(path, window=10):
        if not os.path.exists(path): return None
        df = pd.read_csv(path)
        for c in ['Reward', 'Steps', 'Success', 'Crash']: 
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df['Reward_Smooth'] = df['Reward'].rolling(window=window).mean()
        df['Steps_Smooth'] = df['Steps'].rolling(window=window).mean()
        return df

    df_her = load_and_smooth(her_path)
    df_no = load_and_smooth(no_her_path)

    # GRAPH 1: REWARD COMPARISON (HER vs Baseline)
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if df_her is not None:
        plt.plot(df_her['Episode'], df_her['Reward_Smooth'], color='blue', linewidth=2, label='DDPG + HER')
        plt.fill_between(df_her['Episode'], df_her['Reward_Smooth'], alpha=0.1, color='blue')
        
    if df_no is not None:
        plt.plot(df_no['Episode'], df_no['Reward_Smooth'], color='red', linewidth=2, linestyle='--', label='Baseline (No HER)')
    
    plt.title("Reward Convergence: HER vs Baseline", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Average Reward (Smoothed)", fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.savefig(os.path.join(RESULTS_DIR, "compare_reward.png"), dpi=300)
    print("Saved: compare_reward.png")

    # GRAPH 2: EFFICIENCY (STEPS) COMPARISON
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if df_her is not None:
        plt.plot(df_her['Episode'], df_her['Steps_Smooth'], color='blue', linewidth=2, label='DDPG + HER')
        
    if df_no is not None:
        plt.plot(df_no['Episode'], df_no['Steps_Smooth'], color='red', linewidth=2, linestyle='--', label='Baseline')

    plt.title("Parking Efficiency: Steps to Finish", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Avg Steps", fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.savefig(os.path.join(RESULTS_DIR, "compare_efficiency.png"), dpi=300)
    print("Saved: compare_efficiency.png")

    # GRAPH 3: BEHAVIOR PROFILE (HER AGENT ONLY)
    # We only show this for the HER agent to explain "How it learned"
    if df_her is not None:
        plt.figure(figsize=(10, 6))
        
        scale = 100 if df_her['Success'].max() <= 1.0 else 1
        win = 10
        
        s = df_her['Success'].rolling(window=win, min_periods=1).mean() * scale
        c = df_her['Crash'].rolling(window=win, min_periods=1).mean() * scale
        t = 100 - (s + c)
        t = t.clip(lower=0)

        plt.stackplot(df_her['Episode'], s, c, t,
                      labels=['Success (Parked)', 'Crash', 'Timeout/ Exploring'],
                      colors=['#2ecc71', '#e74c3c', '#bdc3c7'], alpha=0.85)
        
        plt.title("Learning Phases (HER Agent)", fontsize=16)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Outcome Probability (%)", fontsize=14)
        plt.legend(loc='center right')
        plt.margins(0,0)
        plt.savefig(os.path.join(RESULTS_DIR, "behavior_profile_her.png"), dpi=300)
        print("Saved: behavior_profile_her.png")
    
    plt.show()

def compare_three_agents():
    """
    Runs a performance test for 100 episodes for 3 Agents:
    1. Random Agent (Baseline)
    2. DDPG
    3. DDPG with HER
    
    """
    
    print("="*60)
    print("FINAL COMPARATIVE TEST: Random vs No-HER vs HER")
    print("Running 100 episodes for each agent...")
    print("\nThis may take several minutes...")
    print("="*60)
    
    n_episodes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define Agents
    models = [
        {"name": "Random",      "file": None,                     "color": "gray"},
        {"name": "DDPG (Base)", "file": "ddpg_agent_no_her.pth",  "color": "red"},
        {"name": "DDPG + HER",  "file": "HIGH_SUCCESS_success_100.0_crash_0.0_20260104_123740.pth",   "color": "blue"}
    ]
    
    results = []
    
    env = ParallelParkingEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = np.array([np.pi / 4, 1.0])

    print(f"{'Agent':<15} | {'Success':<8} | {'Crash':<8} | {'Steps':<8}")
    print("-" * 50)

    for model_config in models:
        name = model_config["name"]
        filename = model_config["file"]
        
        agent = DDPGAgent(state_dim, action_dim, max_action, device=device)
        
        if filename:
            path = os.path.join(RESULTS_DIR, filename)
            if os.path.exists(path):
                agent.load(path)
            else:
                print(f"Error: {filename} not found. Skipping {name}.")
                continue
        
        success_count = 0
        crash_count = 0
        total_steps = 0
        
        # Run Exam
        for i in range(n_episodes):
            obs, info = env.reset(seed=42 + i) 
            done = False
            ep_steps = 0
            
            while not done:
                if filename is None:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs, noise=False)
                
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_steps += 1
            
            if info.get('is_success', False): success_count += 1
            if info.get('is_crash', False) or info.get('is_out_of_bounds', False): crash_count += 1
            total_steps += ep_steps

        success_rate = (success_count / n_episodes) * 100
        crash_rate = (crash_count / n_episodes) * 100
        avg_steps = total_steps / n_episodes
        
        print(f"{name:<15} | {success_rate:5.1f}%  | {crash_rate:5.1f}%  | {avg_steps:5.1f}")
        
        results.append({
            "Agent": name,
            "Success": success_rate,
            "Crash": crash_rate,
            "Steps": avg_steps,
            "Color": model_config["color"]
        })

    env.close()

    if not results: 
        return

    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Success Rate
    axes[0].bar(df['Agent'], df['Success'], color=df['Color'], alpha=0.7, edgecolor='black')
    axes[0].set_title("Success Rate (Higher is Better)")
    axes[0].set_ylabel("Percent (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Crash Rate
    axes[1].bar(df['Agent'], df['Crash'], color=df['Color'], alpha=0.7, edgecolor='black')
    axes[1].set_title("Crash Rate (Lower is Better)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', alpha=0.3)

    # 3. Average Steps
    axes[2].bar(df['Agent'], df['Steps'], color=df['Color'], alpha=0.7, edgecolor='black')
    axes[2].set_title("Average Steps (Lower is Better)")
    axes[2].set_ylabel("Steps")
    axes[2].grid(axis='y', alpha=0.3)
    
    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontweight='bold')

    plt.suptitle(f"Agent Performance Comparison (n={n_episodes} Episodes)", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, "final_comparison_metrics.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved comparison chart to: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("Pick training mode:")
    print("1. Train DDPG + HER: Train the main agent using Hindsight Experience Replay (HER).")
    print("2. Resume DDPG + HER: Continue training the HER agent from a previous checkpoint.")
    print("3. Train Baseline No HER: Train the agent without HER (baseline).")
    print("4. Quick visual test: Run a single episode demonstration to verify the environment.")
    print("5. Test saved model: Visually evaluate a pre-trained agent.")
    print("6. Plot Comparison Graph: Generate a plot comparing DDPG+HER and DDPG performance.")
    print("7. Plot Additional Metrics from HER Training Log: Generate detailed plots from HER log.")
    print("8. Plot final Comparison: Random vs No-HER vs HER (3 agents, 3 bar charts).")

    choice = input("Enter choice (1-8): ").strip()

    if choice == "1":
        rendering = input("Enable visualisation? (y/n): ").strip().lower() == 'y'
        visual_training_demo(resume=False, use_her=True, rendering=rendering)
    elif choice == "2":
        rendering = input("Enable visualisation? (y/n): ").strip().lower() == 'y'
        visual_training_demo(resume=True, use_her=True, rendering=rendering)
    elif choice == "3":
        rendering = input("Enable visualisation? (y/n): ").strip().lower() == 'y'
        visual_training_demo(resume=False, use_her=False, rendering=rendering)
    elif choice == "4":
        quick_visual_test()
    elif choice == "5":
        model_input = input("Enter model path: ").strip()
        if not model_input.endswith(".pth"): model_input += ".pth"
        rendering = input("Enable visualisation? (y/n): ").strip().lower() == 'y'
        test_model(os.path.join(RESULTS_DIR, model_input), rendering=rendering)
    elif choice == "6":
        plot_training_results()
    elif choice == "7":
        plot_detailed_comparisons()
    elif choice == "8":
        compare_three_agents()
    else:
        print("Invalid choice.")
