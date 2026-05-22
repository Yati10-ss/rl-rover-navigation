# 🚀 Curiosity-Driven RL for Autonomous Rover Navigation on Sparse Lunar Terrains

A curiosity-driven Deep Reinforcement Learning framework that combines **Proximal Policy Optimization (PPO)** with an **Intrinsic Curiosity Module (ICM)** to enable autonomous rover navigation on procedurally generated sparse-reward lunar terrains — achieving a **67.4% goal success rate** vs. a **25% baseline PPO agent**.

> Developed as part of the CS-617 Reinforcement Learning course, MS in Artificial Intelligence & Machine Learning — Drexel University (2025).

---

## 🎯 Problem Statement

In planetary exploration missions on the Moon or Mars, autonomous rovers must navigate vast, unmapped terrains with **no GPS, no prior maps, and minimal real-time communication**. Traditional RL methods fail in these conditions because extrinsic rewards are extremely sparse — science goals may only be reached after hundreds of sequential actions.

This project addresses that challenge by equipping a rover agent with **intrinsic curiosity** — an internal reward signal based on prediction error — enabling self-directed exploration without relying on frequent external feedback.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│              PPO + ICM Agent                        │
│                                                     │
│  Observation: [x, y, θ, velocity, LIDAR, terrain]  │
│                        │                            │
│              ┌─────────▼──────────┐                │
│              │   Shared Encoder   │                │
│              │  (Conv + FC layers)│                │
│              └────┬───────────┬───┘                │
│                   │           │                     │
│         ┌─────────▼──┐  ┌────▼──────────────┐     │
│         │ PPO Policy │  │ Intrinsic Curiosity│     │
│         │  (Actor-   │  │ Module (ICM)       │     │
│         │  Critic)   │  │ Forward + Inverse  │     │
│         └─────────┬──┘  └────┬───────────────┘     │
│                   │          │                      │
│              ┌────▼──────────▼────┐                │
│              │   Hybrid Reward    │                │
│              │ r = r_ext + η·r_int│                │
│              └────────────────────┘                │
└─────────────────────────────────────────────────────┘
                        │
               ┌────────▼────────┐
               │ PyBullet Lunar  │
               │  Environment    │
               │ (Procedural     │
               │  Terrain + Rover│
               │  Dynamics)      │
               └─────────────────┘
```

---

## 📊 Results

### Performance Comparison: PPO vs PPO+ICM

| Metric | PPO (Baseline) | PPO + ICM | Improvement |
|--------|---------------|-----------|-------------|
| Cumulative Extrinsic Reward | 10.2 | **20.8** | +104% |
| Cumulative Intrinsic Reward | — | **12.1** | — |
| Goal Success Rate | 25.0% | **67.4%** | +169% |
| Exploration Coverage | Low | **High** | — |
| ICM Loss (Final Avg) | — | **0.035** | Stable |

### Key Findings
- 🏆 **67.4% goal success rate** — vs 25% for standard PPO — demonstrating the power of intrinsic motivation in sparse-reward environments
- 📈 **ICM loss stabilized at ~0.035** across 5,000 training episodes — confirming robust internal state representation learning
- 🗺️ **Significantly broader exploration coverage** — curiosity-driven agent discovered novel terrain regions the baseline agent never reached
- ⚡ **2x cumulative reward improvement** — intrinsic motivation enabled more efficient policy learning even with minimal extrinsic feedback

---

## 🔬 Methodology

### Environment
- **Physics Engine:** PyBullet with realistic rover dynamics (wheel friction, mass constraints, torque)
- **Terrain:** Procedurally generated using Perlin noise and smoothed elevation profiles inspired by NASA Moon Trek DEMs
- **Obstacles:** Randomly placed rocks and craters simulating real lunar hazards
- **Rover Model:** R2D2 URDF as rover placeholder with continuous control

### State & Action Space
```
State Space:
  - Rover position (x, y, θ)
  - Velocity vectors
  - Simulated LIDAR inputs
  - Terrain gradients

Action Space (Continuous):
  - a1: Forward velocity [0, max_speed]
  - a2: Steering angle [-1, 1]
```

### Reward Formulation
```
r_t = r_ext_t + η · r_int_t

Extrinsic reward:
  +1.0  → goal reached
  -0.01 → per timestep (discourages idling)
   0    → otherwise

Intrinsic reward (ICM):
  r_int_t = ||φ(s_{t+1}) - f(φ(s_t), a_t)||²₂
  (prediction error in latent feature space)
```

### Agent Architecture
- **Policy Network:** PPO Actor-Critic with 2 hidden layers (256 → 128 neurons, ReLU)
- **ICM Forward Model:** 2-layer FC network predicting next state embeddings
- **ICM Inverse Model:** 2-layer FC network predicting actions from state transitions
- **Optimizer:** Adam across all components

### Training Configuration
```
Episodes:           5,000
Max timesteps/ep:   200
PPO update freq:    every 2,048 steps
GAE lambda (λ):     0.95
Discount (γ):       0.99
Clip parameter (ε): 0.2
Entropy reg (β):    0.01
Training time:      ~6 hours on NVIDIA T4 GPU
```

### Curriculum Learning
Training begins on flat terrain before progressively introducing steeper elevations and obstacles — improving stability and preventing early policy collapse.

---

## 🚀 Key Innovations

**1. Curiosity-Augmented PPO for Lunar Navigation**
First implementation combining ICM-based curiosity with continuous navigation control in a physics-accurate PyBullet lunar environment — extending curiosity-driven RL beyond gridworlds and video games into realistic robotics.

**2. Hybrid Reward Annealing**
The intrinsic reward coefficient η is annealed over training to encourage early exploration and later goal-directed exploitation — preventing curiosity from dominating once the terrain is mapped.

**3. Curriculum Terrain Initialization**
Progressive terrain complexity — flat → moderate elevation → cratered — enables stable policy bootstrapping before exposing the agent to full terrain difficulty.

**4. Visual Diagnostic Suite**
Comprehensive visualization pipeline including trajectory GIFs, position-velocity heatmaps, ICM loss curves, and reward dynamics — enabling qualitative and quantitative evaluation of exploration behavior.

---

## 📁 Repository Structure

```
rl-rover-navigation/
│
├── notebooks/
│   └── rl_ppo_icm_training.ipynb      # Full PPO+ICM training pipeline
│
├── scripts/
│   ├── trajectory_heatmap.py          # Animated trajectory GIF generator
│   ├── generate_rover_demo.py         # Flat terrain rover simulation video
│   ├── generate_rover_cratered_demo.py# Cratered terrain rover simulation
│   ├── gif_flat_terrain.py            # Flat terrain GIF
│   ├── gif_cratered_terrain.py        # Cratered terrain GIF
│   └── gif_observations.py           # Observation space visualization GIF
│
├── results/
│   ├── icm_loss_curve.png             # ICM prediction error over training
│   ├── intrinsic_vs_extrinsic.png     # Dual reward signal over episodes
│   ├── trajectory_xy.png             # Rover XY path coverage
│   ├── position_velocity_heatmap.png  # Terrain exploration density
│   └── performance_table.png         # PPO vs PPO+ICM comparison table
│
├── outputs/                           # Generated videos and GIFs (gitignored)
│   └── .gitkeep
│
├── docs/
│   ├── final_research_paper.pdf       # Full IEEE-format research paper
│   └── project_proposal.pdf          # Original project proposal
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- GPU recommended (NVIDIA T4 or better for full training)
- PyBullet requires a display for OpenGL rendering (use virtual display on headless servers)

### Installation

```bash
# Clone the repository
git clone https://github.com/Yati10-ss/rl-rover-navigation.git
cd rl-rover-navigation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Training Notebook

```bash
# Option 1: Jupyter locally
jupyter notebook notebooks/rl_ppo_icm_training.ipynb

# Option 2: Google Colab (recommended for GPU access)
# Upload notebook to colab.research.google.com
```

### Running Visualization Scripts

```bash
# Generate trajectory heatmap GIF
python scripts/trajectory_heatmap.py

# Generate flat terrain simulation
python scripts/generate_rover_demo.py

# Generate cratered terrain simulation
python scripts/generate_rover_cratered_demo.py
```

> ⚠️ **GPU Note:** Simulation scripts use `p.ER_BULLET_HARDWARE_OPENGL` renderer which requires a physical GPU. For headless/CPU environments, replace with `p.ER_TINY_RENDERER` in each script.

---

## 📦 Requirements

```
torch>=2.0.0
pybullet>=3.2.5
stable-baselines3>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
imageio>=2.31.0
opencv-python>=4.8.0
scipy>=1.11.0
gym>=0.26.0
```

---

## 📈 Key Learnings

- **Sparse rewards require intrinsic motivation** — standard PPO stagnates completely in environments where extrinsic feedback arrives only after 200+ sequential actions
- **ICM loss stability validates representation quality** — a decreasing and stable ICM loss (~0.035) confirms the agent is building accurate internal models of state transitions
- **Curriculum learning is critical for stability** — directly training on complex cratered terrain causes early policy collapse; progressive difficulty enables stable bootstrapping
- **Reward annealing prevents over-exploration** — without annealing η, curiosity dominates even after sufficient terrain exploration, preventing goal convergence
- **Physics simulation quality matters** — realistic rover dynamics (friction, torque, mass) in PyBullet forced the agent to learn physically plausible navigation rather than exploiting simulation artifacts

---

## 🔮 Future Work

- **Sim-to-real transfer** using domain randomization — introduce noise, sensor dropouts, and terrain perturbations during training
- **Recurrent policy networks** (LSTM/Transformer) for partial observability and memory-based decision making
- **Multi-agent coordination** — parallel rover exploration with shared curiosity signals and cooperative terrain mapping
- **Alternative intrinsic signals** — comparison with RND, epistemic uncertainty, and empowerment-based exploration
- **Hierarchical RL** — separate high-level exploration goals from low-level motor control for longer-horizon planning
- **Real NASA DEM integration** — replace procedural terrain with actual Moon Trek elevation data

---

## 👥 Contributors

| Contributor | Role & Contributions |
|-------------|----------------------|
| **Yateen Sakhare** | Full code implementation — PPO+ICM training pipeline, Intrinsic Curiosity Module (ICM), PyBullet environment setup, rover dynamics, reward formulation, all visualization scripts (trajectory, heatmap, simulation GIFs/videos), hyperparameter tuning |
| **Chhattu Roy** | Research, literature review, methodology writing, evaluation analysis |
| **Rajni Gandha** | Research, literature review, methodology writing, evaluation analysis |

---

## 📚 References

1. Pathak et al. (2017). Curiosity-driven exploration by self-supervised prediction. CVPRW/ICML.
2. Burda et al. (2018). Exploration by random network distillation. arXiv:1810.12894.
3. Schulman et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
4. Hu et al. (2021). Large-scale autonomous navigation of lunar rover via deep RL. CAC 2021.
5. Zhelo et al. (2018). Curiosity-driven exploration for mapless navigation with DRL. arXiv:1804.00456.
6. Coumans & Bai (2020). PyBullet physics simulation. https://pybullet.org
7. NASA Moon Trek: https://trek.nasa.gov/moon/

---

## 🏫 Academic Context

> Developed for **CS-617-001 Reinforcement Learning**, MS Artificial Intelligence & Machine Learning program, **Drexel University** (Spring 2025).
