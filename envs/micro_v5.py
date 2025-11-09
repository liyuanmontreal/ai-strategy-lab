import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
import cv2

MOVE_ACTIONS = {
    1: (0, -1), 2: (0, 1),
    3: (-1, 0), 4: (1, 0),
}
ATTACK_ACTIONS = {
    5: (0, -1), 6: (0, 1),
    7: (-1, 0), 8: (1, 0),
}

UNIT_TYPES = {
    "melee":  {"hp": 4, "atk": 2, "range": 1, "heal": 0},
    "ranged": {"hp": 3, "atk": 2, "range": 3, "heal": 0},
    "healer": {"hp": 3, "atk": 0.3, "range": 1, "heal": 0.5},
}


class MicroSkirmishV7(ParallelEnv):
    metadata = {"render_modes": ["ansi", "rgb_array"], "name": "micro_v7"}

    def __init__(self, grid_size=15, n_per_team=3, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.n_per_team = n_per_team
        self.rng = np.random.default_rng(seed)
        self.max_steps = 500
        self.step_count = 0
        self.no_damage_steps = 0
        self.UNIT_TYPES = UNIT_TYPES

        self.pos, self.hp, self.alive, self.team, self.agent_type = {}, {}, {}, {}, {}
        self.agents = []
        self.action_spaces, self.observation_spaces = {}, {}

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.no_damage_steps = 0
        self.pos.clear(), self.hp.clear(), self.alive.clear()
        self.team.clear(), self.agent_type.clear()
        self.agents = []

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
                self.pos[a_name] = (x_start, y)

        for a in self.agents:
            self.action_spaces[a] = spaces.Discrete(9)
            self.observation_spaces[a] = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 3))

        return self.observe_all(), {}

    # =====================
    # ⚔️ Step 逻辑
    # =====================
    def step(self, actions):
        self.step_count += 1
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        # --- 移动 ---
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

        # --- 攻击 ---
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

        # --- 治疗 ---
        for a in self.agents:
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

        # --- 应用伤害 ---
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

        if damage_done:
            self.no_damage_steps = 0
        else:
            self.no_damage_steps += 1

        # --- 终止条件 ---
        red_alive = any(self.alive[a] for a in self.agents if self.team[a] == "red")
        blue_alive = any(self.alive[a] for a in self.agents if self.team[a] == "blue")
        if (
            not red_alive or not blue_alive
            or self.no_damage_steps >= 25
            or self.step_count >= self.max_steps
        ):
            for a in self.agents:
                terminations[a] = True

        return self.observe_all(), rewards, terminations, truncations, {}

    # =====================
    # 🎨 渲染：区分职业颜色+形状
    # =====================
    def render_rgb(self, cell_size=25):
        BG_COLOR = np.array([235, 235, 235], dtype=np.uint8)
        img = np.ones((self.grid_size, self.grid_size, 3), dtype=np.uint8) * BG_COLOR

        big_img = np.kron(img, np.ones((cell_size, cell_size, 1), dtype=np.uint8))

        for a in self.agents:
            if not self.alive[a]:
                continue

            x, y = self.pos[a]
            cx, cy = int((x + 0.5) * cell_size), int((y + 0.5) * cell_size)

            # 阵营颜色
            if self.agent_type[a] == "healer":
                color = (0, 255, 255) if self.team[a] == "blue" else (255, 255, 0)
            elif self.agent_type[a] == "ranged":
                color = (100, 200, 255) if self.team[a] == "blue" else (255, 165, 0)
            else:
                color = (80, 120, 255) if self.team[a] == "blue" else (255, 80, 80)

            # 绘制形状：近战=方形，远程=三角形，治疗=圆形
            if self.agent_type[a] == "melee":
                cv2.rectangle(big_img, (cx - 6, cy - 6), (cx + 6, cy + 6), color, -1)
            elif self.agent_type[a] == "ranged":
                pts = np.array([[cx, cy - 7], [cx - 6, cy + 6], [cx + 6, cy + 6]], np.int32)
                cv2.fillConvexPoly(big_img, pts, color)
            elif self.agent_type[a] == "healer":
                cv2.circle(big_img, (cx, cy), 6, color, -1)

            # 血条
            hp = self.hp[a]
            max_hp = UNIT_TYPES[self.agent_type[a]]["hp"]
            ratio = hp / max_hp
            bar_w = 14
            bar_h = 3
            bar_x0 = cx - bar_w // 2
            bar_y0 = cy - cell_size // 2 - 5
            color_bar = (0, int(200 * ratio + 55), int(50 * (1 - ratio)))
            cv2.rectangle(big_img, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (80, 80, 80), -1)
            cv2.rectangle(big_img, (bar_x0, bar_y0), (bar_x0 + int(bar_w * ratio), bar_y0 + bar_h), color_bar, -1)

        return big_img

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


def env(grid_size=15, n_per_team=3, seed=None):
    return MicroSkirmishV7(grid_size=grid_size, n_per_team=n_per_team, seed=seed)
