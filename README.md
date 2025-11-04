# 🧠 AI Strategy Lab | Mini Multi-Agent Battle Simulator

A lightweight **multi-agent tactical combat environment** designed for research in:

- Multi-Agent Reinforcement Learning (MARL)
- Tactical decision-making & emergent strategy
- Micro-control in RTS-style combat
- Agent cooperation & adversarial behavior

Inspired by **StarCraft micro battles**, tactical board games, and lightweight RTS AI systems — simple enough to train quickly, yet expressive enough to study strategy.

---

## ✨ Key Features

| Capability | Description |
|---|---|
Multi-agent adversarial environment | Red vs Blue combat squads  
Action model | Movement, attack, damage, death, victory  
Training support | Rule-based agents & PPO (SB3)  
Rendering | ASCII + GIF replay system  
Frameworks | PettingZoo + SuperSuit + Stable-Baselines3  
Extensibility | Ranged units, fog-of-war, terrain, LLM commander  

---

## 🎮 Environment Overview

### 👥 Agents
Two teams:

- `red_0 ... red_N`
- `blue_0 ... blue_N`

### 🎯 Action Space (9 discrete actions)

| ID | Action |
|---|---|
0 | Stay  
1 | Move up  
2 | Move down  
3 | Move left  
4 | Move right  
5 | Attack up  
6 | Attack down  
7 | Attack left  
8 | Attack right  

### ❤️ Stats

| Attribute | Default |
|---|---|
HP | 3  
Grid | 15×15 (configurable)  
Rewards | +1 kill, shaped micro-rewards  

### 🔄 Observation

RGB grid representation:

- Friend channel  
- Enemy channel  
- Self-position channel  

---

## 📂 Project Structure
envs/
└── micro_v1.py # Core battle environment
utils/
└── replay_recorder.py # GIF replay recorder
train/
└── train_sb3_ppo.py # PPO training script
baselines/
└── rule_based.py # Rule-based benchmark

Environment entry point:

```python
from envs.micro_v1 import env
e = env(grid_size=15, n_per_team=5)

🚀 Getting Started
Install
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

Run rule-based agents
python baselines/rule_based.py

Train PPO agent
python train/train_sb3_ppo.py --grid-size 12 --units 4 --num-envs 2 --total-steps 200000

🎥 Generate Battle Replay GIF
python utils/replay_recorder.py --steps 400 --outfile fight.gif


Example:

Units advance, engage, take damage, die, and one side wins.

🧠 Research Directions

Micro tactics: focus-fire, flanking, kiting

Emergent teamwork & coordination

LLM-guided reinforcement learning (“AI Commander”)

Strategy curriculum & self-play evolution

Partial observability (fog-of-war)

🧩 Roadmap
Version	Feature
✅ v1	Movement, melee combat, replay, PPO
🟡 v2	Fog-of-war, vision, recurrent PPO (LSTM)
🟡 v3	Unit types: ranged / melee / healer
⬜ v4	Terrain, cover, obstacles
⬜ v5	Resource & build system
⬜ v6	LLM tactical commander (high-level planning)
🛠️ Tech Stack
Category	Tools
RL	Stable-Baselines3 (PPO)
Multi-Agent Env	PettingZoo ParallelEnv
Vectorization	SuperSuit
Visualization	ASCII → GIF (pygame WIP)
Logging (optional)	TensorBoard / Weights & Biases
🤝 Acknowledgements

PettingZoo

Stable-Baselines3

StarCraft AI research community

Multi-Agent RL literature

📬 Contact

Interested in:

RL / MARL

RTS AI experiments

Game-AI research

Lightweight custom environments

Let’s connect!

