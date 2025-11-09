import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces

# 动作定义
MOVE_ACTIONS = {
    1: (0, -1),  # up
    2: (0, 1),   # down
    3: (-1, 0),  # left
    4: (1, 0),   # right
}
ATTACK_ACTIONS = {
    5: (0, -1),  # atk up
    6: (0, 1),   # atk down
    7: (-1, 0),  # atk left
    8: (1, 0),   # atk right
}
ALL_ACTIONS = list(range(9))

# 兵种参数
UNIT_TYPES = {
    "melee":  {"hp": 4, "atk": 2, "range": 1, "heal": 0},
    "ranged": {"hp": 3, "atk": 2, "range": 3, "heal": 0},
    "healer": {"hp": 3, "atk": 0, "range": 1, "heal": 0.5},
}


class MicroSkirmishV2(ParallelEnv):
    metadata = {"name": "micro_skirmish_v2", "render_modes": ["human", "ansi"]}

    def __init__(
        self,
        grid_size: int = 15,
        max_steps: int = 200,
        seed: int = 42,
        # 每队兵种配置：[("type", count), ...]
        team_comp=None,
        vision_radius: int = 4,
        
    ):
        """
        team_comp 例如:
        [("melee", 2), ("ranged", 2), ("healer", 1)]
        """
        if team_comp is None:
            team_comp = [("melee", 2), ("ranged", 2), ("healer", 1)]

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.vision_radius = vision_radius
        self.rng = np.random.RandomState(seed)
        self.render_mode = "ansi"    # 默认渲染模式

        # 展开兵种配置
        self.team_comp = team_comp
        self.possible_agents = []
        self.agent_type = {}
        self.team = {}

        def add_team(prefix):
            idx = 0
            for utype, cnt in team_comp:
                for _ in range(cnt):
                    name = f"{prefix}_{idx}"
                    self.possible_agents.append(name)
                    self.agent_type[name] = utype
                    self.team[name] = prefix
                    idx += 1

        add_team("red")
        add_team("blue")

        self.agents = []

        # observation: (patch, patch, channels)
        # channels: [ally_presence, enemy_presence, ally_hp, enemy_hp, self_mask]
        patch = 2 * vision_radius + 1
        self._obs_shape = (patch, patch, 5)

        self.observation_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=self._obs_shape, dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(len(ALL_ACTIONS)) for a in self.possible_agents
        }

        # state vars
        self.pos = {}
        self.hp = {}
        self.alive = {}
        self.steps = 0

    # PettingZoo API
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.agents = self.possible_agents[:]
        self.steps = 0
        self.pos = {}
        self.hp = {}
        self.alive = {a: True for a in self.agents}

        # 布阵：红左蓝右，稍微错位避免完全对称卡死
        # 红队从左往右，蓝队从右往左
        def spawn_team(prefix, x_col):
            row = 2
            for a in self.agents:
                if self.team[a] != prefix:
                    continue
                utype = self.agent_type[a]
                self.pos[a] = (x_col, row)
                self.hp[a] = UNIT_TYPES[utype]["hp"]
                row += 2
                if row >= self.grid_size - 2:
                    row = 2
                    x_shift = -1 if prefix == "red" else 1
                    x_col += x_shift

        spawn_team("red", 2)
        spawn_team("blue", self.grid_size - 3)

        obs = self._get_obs_all()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        # 若无人存活则不应再 step
        if len(self.agents) == 0:
            raise RuntimeError("step() called after all agents removed")

        self.steps += 1

        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # ---- 1) 移动意图 ----
        desired = {}
        for a in self.agents:
            if not self.alive.get(a, False):
                continue
            act = int(actions.get(a, 0))
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

        # ---- 2) 碰撞处理（简单：同格冲突则都不动）----
        new_pos = self.pos.copy()
        cell2agents = {}
        for a, p in desired.items():
            if not self.alive[a]:
                continue
            cell2agents.setdefault(p, []).append(a)

        for cell, lst in cell2agents.items():
            if len(lst) == 1:
                new_pos[lst[0]] = cell
            else:
                # 简单处理：全部保持原位（可以后续改 swap/优先级）
                pass

        self.pos = new_pos

        # ---- 3) 攻击 & 治疗 意图收集 ----
        dmg_to = {a: 0 for a in self.agents}
        heal_to = {a: 0 for a in self.agents}

        for a in self.agents:
            if not self.alive[a]:
                continue
            act = int(actions.get(a, 0))
            utype = self.agent_type[a]
            cfg = UNIT_TYPES[utype]

            if act in ATTACK_ACTIONS:
                dx, dy = ATTACK_ACTIONS[act]
                ax, ay = self.pos[a]
                rng = cfg["range"]

                # 扫描该方向上的格子(1..range)
                tx, ty = ax, ay
                target_found = None
                for _ in range(rng):
                    tx += dx
                    ty += dy
                    if not (0 <= tx < self.grid_size and 0 <= ty < self.grid_size):
                        break
                    # 找到第一个单位就停
                    for b in self.agents:
                        if not self.alive.get(b, False):
                            continue
                        if self.pos[b] == (tx, ty):
                            target_found = b
                            break
                    if target_found is not None:
                        break

                if target_found is not None:
                    # 治疗兵：给友军加血
                    if cfg["heal"] > 0 and self.team[target_found] == self.team[a]:
                        heal_to[target_found] += cfg["heal"]
                        rewards[a] += 0.02
                    # 其他：攻击敌人
                    elif cfg["atk"] > 0 and self.team[target_found] != self.team[a]:
                        dmg_to[target_found] += cfg["atk"]
                        rewards[a] += 0.05

        # ---- 4) 结算伤害 & 治疗 ----
        for a in self.agents:
            if not self.alive[a]:
                continue

            # 治疗
            if heal_to[a] > 0:
                max_hp = UNIT_TYPES[self.agent_type[a]]["hp"]
                self.hp[a] = float(min(max_hp, self.hp[a] + heal_to[a]))

            # 伤害
            if dmg_to[a] > 0:
                self.hp[a] -= float(dmg_to[a])
                if self.hp[a] <= 0:
                    self.alive[a] = False
                    # 队友整体小罚
                    for x in self.agents:
                        if self.alive.get(x, False) and self.team[x] == self.team[a]:
                            rewards[x] -= 0.1
                    # 敌方小奖励
                    for x in self.agents:
                        if self.alive.get(x, False) and self.team[x] != self.team[a]:
                            rewards[x] += 0.2

        # ---- 5) 检查结束条件 ----
        red_alive = any(self.alive[a] for a in self.agents if self.team[a] == "red")
        blue_alive = any(self.alive[a] for a in self.agents if self.team[a] == "blue")

        done = False
        if not red_alive or not blue_alive:
            done = True
        if self.steps >= self.max_steps:
            # 平局：不给额外奖励
            done = True
            for a in self.agents:
                truncations[a] = True

        if done:
            for a in self.agents:
                terminations[a] = True

        obs = self._get_obs_all()
        return obs, rewards, terminations, truncations, infos

    # ---- 观测：局部视野 patch ----
    def _get_obs_all(self):
        return {a: self._get_obs(a) for a in self.agents}

    def _get_obs(self, agent):
        patch = self._obs_shape[0]
        r = self.vision_radius
        cx, cy = self.pos[agent]
        team = self.team[agent]

        # 通道: ally, enemy, ally_hp, enemy_hp, self_mask
        obs = np.zeros(self._obs_shape, dtype=np.float32)

        for b in self.agents:
            if not self.alive.get(b, False):
                continue
            x, y = self.pos[b]

            if abs(x - cx) > r or abs(y - cy) > r:
                continue  # 超出视野(方形视野，简单一点)

            px = x - cx + r
            py = y - cy + r
            if not (0 <= px < patch and 0 <= py < patch):
                continue

            hp_norm = self.hp[b] / UNIT_TYPES[self.agent_type[b]]["hp"]

            if self.team[b] == team:
                obs[py, px, 0] = 1.0
                obs[py, px, 2] = hp_norm
            else:
                obs[py, px, 1] = 1.0
                obs[py, px, 3] = hp_norm

        # self mask
        obs[r, r, 4] = 1.0
        return obs

    # ---- 渲染：全图信息（调试用，不受迷雾限制）----
    def render(self, mode="human"):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype="<U1")
        for a in self.agents:
            if not self.alive.get(a, False):
                continue
            x, y = self.pos[a]
            grid[y, x] = "R" if self.team[a] == "red" else "B"
        s = "\n".join("".join(row) for row in grid)
        if mode == "human":
            print(s)
            print()
            return None
        elif mode == "ansi":
            return s
        else:
            raise NotImplementedError

    def close(self):
        return

def env(grid_size=15, max_steps=200, seed=42,
        team_comp=None, vision_radius=4):
    return MicroSkirmishV2(
        grid_size=grid_size,
        max_steps=max_steps,
        seed=seed,
        team_comp=team_comp,
        vision_radius=vision_radius,
    )
