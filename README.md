# Reinforcement Learning for Parallel Parking with DDPG

This project implements a **Deep Deterministic Policy Gradient (DDPG)** agent trained to perform **parallel parking** in a custom parking environment.  
It includes implementations for training with **Hindsight Experience Replay (HER)**, and tools for visualising performance comparisons.

---

## 1. Installation

Ensure you have **Python** installed.  
Navigate to the project root directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **torch**
- **gymnasium**
- **highway-env**
- **numpy**

---

## 2. Running the Code

The core running is located in the `experiments/` folder.  
You can run these scripts directly from the project root.

### A. Headless Training (Fastest)

Use this script for standard training **without visual rendering**. It is optimised for speed.

```bash
python experiments/train_ddpg.py
```

**Menu Options:**

- **Full training:** Trains a new agent for 2000 episodes (no rendering).
- **Test saved model:** Runs 100 test episodes using a specific `.pth` model file.
- **Resume training:** Continues training from a saved checkpoint (500 episodes).

---

### B. Visual Training & Analysis

Use this script to **visualise the parking**, train with/without HER, and generate performance comparison plots.

```bash
python experiments/visual_train_ddpg.py
```

**Menu Options:**

1. **Train DDPG + HER:** Trains the main agent using Hindsight Experience Replay
2. **Quick visual test:** Runs a single episode demo to verify the environment works.
3. **Test saved model:** Visual evaluation of a pre-trained agent.
4. **Resume DDPG + HER:** Continue training the HER agent.

---

## 3. Project Structure

```
ddpg/          # Contains the DDPG agent, Actor/Critic networks, noise processes
env/           # Custom ParallelParkingEnv environment wrapper
experiments/   # Main executable scripts for training and testing
results/       # Stores trained models (.pth)
```

See `results/parallel_parking/MODELS.md` for details on the naming convention of saved models.
