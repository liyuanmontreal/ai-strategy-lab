#  AI Strategy Lab | Mini Multi-Agent Battle Simulator

A lightweight **multi-agent tactical combat environment** designed for research in:

- Multi-Agent Reinforcement Learning (MARL)
- Tactical decision-making & emergent strategy
- Micro-control in RTS-style combat
- Agent cooperation & adversarial behavior

Inspired by **StarCraft micro battles**, tactical board games, and lightweight RTS AI systems — simple enough to train quickly, yet expressive enough to study strategy.

---

##  Key Features

| Capability | Description |
|---|---|
Multi-agent adversarial environment | Red vs Blue combat squads  
Action model | Movement, attack, damage, death, victory  
Training support | Rule-based agents & PPO (SB3)  
Rendering | ASCII + GIF replay system  
Frameworks | PettingZoo + SuperSuit + Stable-Baselines3  
Extensibility | Ranged units, fog-of-war, terrain, LLM commander  

---

##  Environment Overview

###  Agents
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

###  Stats

| Attribute | Default |
|---|---|
HP | 3  
Grid | 15×15 (configurable)  
Rewards | +1 kill, shaped micro-rewards  

###  Observation

RGB grid representation:

- Friend channel  
- Enemy channel  
- Self-position channel  

---

##  Project Structure
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

Getting Started
Install
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

Run rule-based agents
python baselines/rule_based.py

Train PPO agent
python train/train_sb3_ppo.py --grid-size 12 --units 4 --num-envs 2 --total-steps 200000


 Generate Battle Replay GIF
python utils/replay_recorder.py --steps 400 --outfile fight.gif

V2 version:
python train/train_rnn_ppo_v2.py --grid-size 15 --num-envs 4 --total-steps 400000
python utils/replay_v2.py --steps 400 --outfile fight_v2.gif

v3
python baselines\rule_based_v3.py --steps 400 --outfile fight_rule_v3.gif


