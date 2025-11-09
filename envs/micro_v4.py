import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces


# =======================
# ⚙️ 全局参数
# =======================
MOVE_ACTIONS = {
    1: (0, -1),  # up
    2: (0, 1),   # down
    3: (-1, 0),  # left
    4: (1, 0),   # right
}

ATTACK_ACTIONS = {
    5: (0, -1),  # up
    6: (0, 1),   # down
    7: (-1, 0),  # left
    8: (1, 0),   # right
}

UNIT_TYPES = {
    "melee":  {"hp": 4, "atk": 2, "range": 1, "heal": 0},
    "ranged": {"hp": 3, "atk": 2, "range": 3, "heal": 0},
    "healer": {"hp": 3, "atk": 0.3, "range": 1, "heal": 0.5},  # 医疗兵可微弱攻击
}


# =======================
# 🧩 环境类
# =======================
class MicroSkirmishV5(ParallelEnv):
    metadata = {"render_modes": ["ansi", "rgb_array"], "name": "micro_v5"}

    def __init__(self, grid_size=15, n_per_team=3, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.n_per_team = n_per_team
        self.rng = np.random.default_rng(seed)
        self.pos, self.hp, self.alive, self.team, self.agent_type = {}, {}, {}, {}, {}
        self.agents = []
        self.max_steps = 500
        self.step_count = 0
        self.no_damage_steps = 0  # ✅ 僵局检测计数器
        self.UNIT_TYPES = UNIT_TYPES

        self.action_spaces = {}
        self.observation_spaces = {}

    # =======================
    # 🔁 重置
    # =======================
    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.no_damage_steps = 0
        self.pos.clear(), self.hp.clear(), self.alive.clear()
        self.team.clear(), self.agent_type.clear()
        self.agents = []

        # 🟥 红队 & 🟦 蓝队
        for i in range(self.n_per_team):
            for team_name, x_start, types in [
                ("red", 1, ["melee", "ranged", "healer"]),
                ("blue", self.grid_size - 2, ["melee", "ranged", "healer"]),
            ]:
                a_name = f"{team_name}_{types[i]}"
                self.agents.append(a_name)
                self.team[a_name] = team_name
                self.agent_type[a_name] = types[i]
                self.hp[a_name] = UNIT_TYPES[types[i]]["hp"]
                self.alive[a_name] = True
                y = self.grid_size // 2 + i - 1
                x = x_start
                self.pos[a_name] = (x, y)

        # 定义空间
        for a in self.agents:
            self.action_spaces[a] = spaces.Discrete(9)
            self.observation_spaces[a] = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 3))

        return self.observe_all(), {}

    # =======================
    # ⚙️ Step 函数
    # =======================
    def step(self, actions):
        self.step_count += 1
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        # ---------- 1️⃣ 移动 ----------
        desired = {}
        for a, act in actions.items():
            if not self.alive[a]:
                continue
            if act in MOVE_ACTIONS:
                dx, dy = MOVE_ACTIONS[act]
                x, y = self.pos[a]
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    desired[a] = (nx, ny)
                else:
                    desired[a] = (x, y)
            else:
                desired[a] = self.pos[a]

        # 碰撞处理
        new_pos = self.pos.copy()
        target2agents = {}
        for a, p in desired.items():
            if not self.alive[a]:
                continue
            target2agents.setdefault(p, []).append(a)
        for p, lst in target2agents.items():
            if len(lst) == 1:
                new_pos[lst[0]] = p
        self.pos = new_pos

        # ---------- 2️⃣ 攻击 ----------
        dmg_to_apply = {a: 0 for a in self.agents if self.alive[a]}
        for a, act in actions.items():
            if not self.alive[a]:
                continue
            if act in ATTACK_ACTIONS:
                ax, ay = self.pos[a]
                atk_val = UNIT_TYPES[self.agent_type[a]]["atk"]
                atk_range = UNIT_TYPES[self.agent_type[a]]["range"]
                for b in self.agents:
                    if not self.alive[b] or self.team[b] == self.team[a]:
                        continue
                    bx, by = self.pos[b]
                    dx, dy = bx - ax, by - ay
                    if max(abs(dx), abs(dy)) <= atk_range:
                        dmg_to_apply[b] += atk_val
                        rewards[a] += 0.05

        # ---------- 3️⃣ 治疗 ----------
        for a, act in actions.items():
            if not self.alive[a]:
                continue
            utype = self.agent_type[a]
            if UNIT_TYPES[utype]["heal"] <= 0:
                continue
            heal_range = UNIT_TYPES[utype]["range"]
            heal_val = UNIT_TYPES[utype]["heal"]
            ax, ay = self.pos[a]
            for b in self.agents:
                if not self.alive[b] or self.team[b] != self.team[a]:
                    continue
                bx, by = self.pos[b]
                if max(abs(bx - ax), abs(by - ay)) <= heal_range:
                    self.hp[b] = min(UNIT_TYPES[self.agent_type[b]]["hp"], self.hp[b] + heal_val)

        # ---------- 4️⃣ 应用伤害 ----------
        damage_done = False
        for b, dmg in dmg_to_apply.items():
            if dmg <= 0:
                continue
            old_hp = self.hp[b]
            self.hp[b] = max(0, old_hp - dmg)
            if self.hp[b] < old_hp:
                damage_done = True
            if self.hp[b] <= 0 and old_hp > 0:
                self.alive[b] = False
                for a in self.agents:
                    if self.alive[a] and self.team[a] != self.team[b]:
                        rewards[a] += 1.0

        # ✅ 更新僵局计数
        if damage_done:
            self.no_damage_steps = 0
        else:
            self.no_damage_steps += 1

        # ---------- 5️⃣ 结束判定 ----------
        red_alive = any(self.alive[a] for a in self.agents if self.team[a] == "red")
        blue_alive = any(self.alive[a] for a in self.agents if self.team[a] == "blue")
        red_attackers = any(
            self.alive[a] and UNIT_TYPES[self.agent_type[a]]["atk"] > 0 and self.team[a] == "red"
            for a in self.agents
        )
        blue_attackers = any(
            self.alive[a] and UNIT_TYPES[self.agent_type[a]]["atk"] > 0 and self.team[a] == "blue"
            for a in self.agents
        )

        if (
            not red_alive
            or not blue_alive
            or not red_attackers
            or not blue_attackers
            or self.no_damage_steps >= 25   # ✅ 连续25回合无伤害 → 僵局结束
            or self.step_count >= self.max_steps
        ):
            for a in self.agents:
                terminations[a] = True

        return self.observe_all(), rewards, terminations, truncations, {}

    # =======================
    # 👁️ 观测
    # =======================
    def observe_all(self):
        obs = {}
        for a in self.agents:
            grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
            for b in self.agents:
                if not self.alive[b]:
                    continue
                x, y = self.pos[b]
                if self.team[b] == "red":
                    grid[y, x, 0] = 1.0
                else:
                    grid[y, x, 2] = 1.0
            obs[a] = grid
        return obs

    def render(self, mode="ansi"):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for a in self.agents:
            if not self.alive[a]:
                continue
            x, y = self.pos[a]
            char = "R" if self.team[a] == "red" else "B"
            grid[y][x] = char
        return "\n".join("".join(row) for row in grid)


# =======================
# 构造函数
# =======================
def env(grid_size=15, n_per_team=3, seed=None):
    return MicroSkirmishV5(grid_size=grid_size, n_per_team=n_per_team, seed=seed)
