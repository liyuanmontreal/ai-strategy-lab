import os
import sys
import numpy as np
import imageio

# ---- 路径修复 ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.micro_v2 import env as make_env, UNIT_TYPES


def rule_based_policy(env, a):
    """为单个智能体定义规则策略：锁定目标 + 八方向攻击 + 修正攻击距离"""
    if not env.alive.get(a, False):
        return 0

    utype = env.agent_type[a]
    ax, ay = env.pos[a]

    # 初始化目标锁定
    if not hasattr(env, "target_lock"):
        env.target_lock = {}

    # 🩹 医疗兵逻辑
    if utype == "healer":
        allies = [b for b in env.agents if env.team[b] == env.team[a] and env.alive[b]]
        injured = [b for b in allies if env.hp[b] < UNIT_TYPES[env.agent_type[b]]["hp"]]
        if injured:
            target = min(injured, key=lambda b: abs(env.pos[b][0]-ax) + abs(env.pos[b][1]-ay))
            ex, ey = env.pos[target]
            dx, dy = ex - ax, ey - ay
            heal_range = UNIT_TYPES[utype]["range"]
            if max(abs(dx), abs(dy)) <= heal_range:
                # 八方向治疗
                if abs(dx) >= abs(dy):
                    return 7 if dx < 0 else 8
                else:
                    return 5 if dy < 0 else 6
            else:
                # 靠近友军
                if abs(dx) > abs(dy):
                    return 3 if dx < 0 else 4
                else:
                    return 1 if dy < 0 else 2
        else:
            return 0  # 没有受伤友军则待机

    # ⚔️ 攻击单位逻辑（近战 / 远程）
    if a not in env.target_lock or not env.alive.get(env.target_lock[a], False):
        enemies = [b for b in env.agents if env.team[b] != env.team[a] and env.alive[b]]
        if not enemies:
            return 0
        env.target_lock[a] = min(enemies, key=lambda b: abs(env.pos[b][0]-ax)+abs(env.pos[b][1]-ay))

    target = env.target_lock[a]
    if not env.alive.get(target, False):
        return 0

    ex, ey = env.pos[target]
    dx, dy = ex - ax, ey - ay
    attack_range = UNIT_TYPES[utype]["range"]

    # ✅ 修正攻击距离（支持八方向）
    if max(abs(dx), abs(dy)) <= attack_range:
        if abs(dx) >= abs(dy):
            return 7 if dx < 0 else 8  # 左右攻击
        else:
            return 5 if dy < 0 else 6  # 上下攻击
    else:
        # 向目标靠近
        if abs(dx) > abs(dy):
            return 3 if dx < 0 else 4
        else:
            return 1 if dy < 0 else 2


def render_frame(env, cell_size=20):
    """渲染带颜色的网格图"""
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

    img = np.kron(grid, np.ones((cell_size, cell_size, 1), dtype=np.uint8))
    return img


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--outfile", type=str, default="fight_rule_v4.gif")
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

    # 胜负判定
    red_alive = sum(env.alive[a] for a in env.agents if env.team[a] == "red")
    blue_alive = sum(env.alive[a] for a in env.agents if env.team[a] == "blue")

    if red_alive == 0 and blue_alive == 0:
        print("🤝 Draw (both sides eliminated)")
    elif red_alive == 0:
        print("💀 Blue wins!")
    elif blue_alive == 0:
        print("❤️ Red wins!")
    else:
        print(f"⏱️ Draw after {args.steps} steps: {red_alive} red vs {blue_alive} blue")

    imageio.mimsave(args.outfile, frames, duration=0.15)
    print(f"✅ Replay saved to {args.outfile}")


if __name__ == "__main__":
    main()
