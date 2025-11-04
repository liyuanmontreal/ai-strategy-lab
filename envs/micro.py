import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces

MOVE_ACTIONS = {
    1: (0, -1),  # up
    2: (0, 1),   # down
    3: (-1, 0),  # left
    4: (1, 0),   # right
}
ATTACK_ACTIONS = {
    5: (0, -1),  # attack up
    6: (0, 1),   # attack down
    7: (-1, 0),  # attack left
    8: (1, 0),   # attack right
}
ALL_ACTIONS = list(range(9))  # 0 = stay

MAX_HP = 3

class MicroSkirmishV1(ParallelEnv):
    
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, grid_size=15, n_per_team=5, max_steps=200, seed=42):
        self.grid_size = grid_size
        self.n_per_team = n_per_team
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)

        self.possible_agents = (
            [f"red_{i}" for i in range(n_per_team)] +
            [f"blue_{i}" for i in range(n_per_team)]
        )
        self.agents = []

        self.action_spaces = {
            a: spaces.Discrete(len(ALL_ACTIONS)) for a in self.possible_agents
        }
        self.observation_spaces = {
            a: spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32)
            for a in self.possible_agents
        }

    def _same_team(self, a, b):
        return a.split("_")[0] == b.split("_")[0]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.agents = self.possible_agents[:]
        self.alive = {a: True for a in self.agents}
        self.hp = {a: MAX_HP for a in self.agents}
        self.steps = 0

        # spawn units: left vs right lines
        self.pos = {}
        for i in range(self.n_per_team):
            # self.pos[f"red_{i}"]  = (2, 2 + i*2)
            # self.pos[f"blue_{i}"] = (self.grid_size-3, 2 + i*2)
            self.pos[f"red_{i}"]  = (2, 2 + i*2)
            self.pos[f"blue_{i}"] = (self.grid_size-3, 3 + i*2)  # +1 offset

        obs = self._get_obs()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _get_obs(self):
        # simple spatial map: channel = friendly, enemy, self-pos
        obs = {}
        grid_friend = np.zeros((self.grid_size, self.grid_size))
        grid_enemy = np.zeros((self.grid_size, self.grid_size))

        for a in self.agents:
            if not self.alive[a]: continue
            x, y = self.pos[a]
            if a.startswith("red"):
                grid_friend[y, x] = 1
            else:
                grid_enemy[y, x] = 1

        for a in self.agents:
            my_grid = np.zeros((self.grid_size, self.grid_size, 3))
            if not self.alive[a]:
                obs[a] = my_grid
                continue

            # red sees friend=ch0 enemy=ch1, blue reversed
            if a.startswith("red"):
                my_grid[:, :, 0] = grid_friend
                my_grid[:, :, 1] = grid_enemy
            else:
                my_grid[:, :, 0] = grid_enemy
                my_grid[:, :, 1] = grid_friend

            x, y = self.pos[a]
            my_grid[y, x, 2] = 1
            obs[a] = my_grid

        return obs

    def step(self, actions):
        rewards = {a: 0.0 for a in self.agents}
        self.steps += 1

        # --- Movement ---
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
        cell2units = {}
        for a, p in desired.items():
            if not self.alive[a]: continue
            cell2units.setdefault(p, []).append(a)

        for p, alive_list in cell2units.items():
            if len(alive_list) == 1:
                new_pos[alive_list[0]] = p

        self.pos = new_pos

        # --- Attack ---
        dmg = {a: 0 for a in self.agents}
        for a, act in actions.items():
            if not self.alive[a]: continue
            if act in ATTACK_ACTIONS:
                dx, dy = ATTACK_ACTIONS[act]
                x, y = self.pos[a]
                tx, ty = x + dx, y + dy
                if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
                    for b in self.agents:
                        if not self.alive[b]: continue
                        if self._same_team(a, b): continue
                        if self.pos[b] == (tx, ty):
                            dmg[b] += 1
                            rewards[a] += 0.05

        # --- Apply dmg ---
        for b, d in dmg.items():
            if d == 0: continue
            self.hp[b] -= d
            if self.hp[b] <= 0:
                self.alive[b] = False
                rewards[b] -= 0.2
                for a in self.agents:
                    if self.alive.get(a, False) and not self._same_team(a, b):
                        rewards[a] += 1.0

        # --- Terminal ---
        red_alive = any(self.alive[a] for a in self.agents if a.startswith("red"))
        blue_alive = any(self.alive[a] for a in self.agents if a.startswith("blue"))

        terms = {a: False for a in self.agents}
        truncs = {a: False for a in self.agents}

        if not red_alive or not blue_alive:
            for a in self.agents:
                terms[a] = True

        if self.steps >= self.max_steps:
            for a in self.agents:
                truncs[a] = True

        obs = self._get_obs()
        infos = {a: {} for a in self.agents}

        return obs, rewards, terms, truncs, infos
    
    def render_frame(env):
        s = env.render("ansi")  # ✅ 必须返回字符串
        lines = s.split("\n")
        img = []
        for line in lines:
            row = []
            for char in line:
                if char == "R":
                    row.append([255, 0, 0])
                elif char == "B":
                    row.append([0, 0, 255])
                else:
                    row.append([255, 255, 255])
            img.append(row)
        return np.array(img, dtype=np.uint8)

    def render(self, mode="human"):
        grid = np.full((self.grid_size, self.grid_size), ".")
        for a in self.agents:
            if not self.alive.get(a, False):
                continue
            x, y = self.pos[a]
            grid[y, x] = "R" if a.startswith("red") else "B"

        s = "\n".join("".join(r) for r in grid)

        if mode == "human":
            print(s)
            print()
            return None
        elif mode == "ansi":
            return s
        else:
            raise NotImplementedError



def env(grid_size=15, n_per_team=5, max_steps=200, seed=42):
    return MicroSkirmishV1(grid_size, n_per_team, max_steps, seed)
