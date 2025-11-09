import os
import sys
import numpy as np
import imageio

# ---- 路径修复：确保可从任意目录运行 ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.micro_v2 import env as make_env, UNIT_TYPES


def rule_based_policy(env, a):
    """为单个智能体定义规则策略：近战/远程攻击 + 治疗行为"""
    if not env.alive.get(a, False):
        return 0

    utype = env.agent_type[a]
    ax, ay = env.pos[a]

    # 🩹 治疗兵逻辑
    if utype == "healer":
        allies = [b for b in env.agents if env.team[b] == env.team[a] and env.alive[b]]
        injured = [b for b in allies if env.hp[b] < UNIT_TYPES[env.agent_type[b]]["hp"]]
        if injured:
            target = min(injured, key=lambda b: abs(env.pos[b][0]-ax) + abs(env.pos[b][1]-ay))
            ex, ey = env.pos[target]
            dx, dy = ex - ax, ey - ay
            heal_range = UNIT_TYPES[utype]["range"]
            if abs(dx) + abs(dy) <= heal_range:
                if abs(dx) > abs(dy):
                    return 7 if dx < 0 else 8
                else:
                    return 5 if dy < 0 else 6
            else:
                if abs(dx) > abs(dy):
                    return 3 if dx < 0 else 4
                else:
                    return 1 if dy < 0 else 2
        else:
            # 无受伤队友：靠近友军
            if allies:
                buddy = min(allies, key=lambda b: abs(env.pos[b][0]-ax) + abs(env.pos[b][1]-ay))
                bx, by = env.pos[buddy]
                if abs(bx-ax)+abs(by-ay) > 2:
                    if abs(bx-ax) > abs(by-ay):
                        return 3 if bx < ax else 4
                    else:
                        return 1 if by < ay else 2
            return 0

    # ⚔️ 攻击单位逻辑（近战 / 远程）
    enemies = [b for b in env.agents if env.team[b] != env.team[a] and env.alive[b]]
    if not enemies:
        return 0

    target = min(enemies, key=lambda b: abs(env.pos[b][0] - ax) + abs(env.pos[b][1] - ay))
    ex, ey = env.pos[target]
    dx, dy = ex - ax, ey - ay
    attack_range = UNIT_TYPES[utype]["range"]

    # 在射程内 → 攻击
    if abs(dx) + abs(dy) <= attack_range:
        if abs(dx) > abs(dy):
            return 7 if dx < 0 else 8
        else:
            return 5 if dy < 0 else 6
    else:
        # 不在射程内 → 移动
        if abs(dx) > abs(dy):
            return 3 if dx < 0 else 4
        else:
            return 1 if dy < 0 else 2


# ---- 渲染工具 ----
def render_frame(env, cell_size=20):
    """把网格状态渲染成彩色图像（红蓝区分 + 兵种颜色）"""
    COLOR_MAP = {
        "red_melee": [255, 60, 60],
        "red_ranged": [255, 150, 100],
        "red_healer": [255, 255, 100],
        "blue_melee": [60, 60, 255],
        "blue_ranged": [100, 150, 255],
        "blue_healer": [120, 255, 255],
    }
    BG_COLOR = np.array([230, 230, 230], dtype=np.uint8)

    grid = np.ones((env.grid_size, env.grid_size, 3), dtype=np.uint8) * BG_COLOR

    for a in env.agents:
        if not env.alive.get(a, False):
            continue
        x, y = env.pos[a]
        key = f"{env.team[a]}_{env.agent_type[a]}"
        color = np.array(COLOR_MAP.get(key, [0, 0, 0]), dtype=np.uint8)
        grid[y, x] = color

    # 放大像素格
    img = np.kron(grid, np.ones((cell_size, cell_size, 1), dtype=np.uint8))
    return img


# ---- 主逻辑 ----
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--outfile", type=str, default="fight_rule_v2.gif")
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env(grid_size=args.grid_size, seed=args.seed)
    obs, info = env.reset()
    frames = []

    for step in range(args.steps):
        actions = {}
        for a in env.agents:
            if not env.alive.get(a, False):
                continue
            actions[a] = rule_based_policy(env, a)

        obs, rew, term, trunc, info = env.step(actions)
        frames.append(render_frame(env))
        if all(term.values()):
            break

    imageio.mimsave(args.outfile, frames, duration=0.15)
    print(f"✅ Rule-based replay saved to {args.outfile}")


if __name__ == "__main__":
    main()
