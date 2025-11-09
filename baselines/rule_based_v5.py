import os
import sys
import numpy as np
import imageio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.micro_v5 import env as make_env, UNIT_TYPES


def rule_based_policy(env, a):
    if not env.alive.get(a, False):
        return 0

    utype = env.agent_type[a]
    ax, ay = env.pos[a]

    if not hasattr(env, "target_lock"):
        env.target_lock = {}

    # 医疗兵逻辑
    if utype == "healer":
        allies = [b for b in env.agents if env.team[b] == env.team[a] and env.alive[b]]
        injured = [b for b in allies if env.hp[b] < UNIT_TYPES[env.agent_type[b]]["hp"]]
        if injured:
            target = min(injured, key=lambda b: abs(env.pos[b][0]-ax) + abs(env.pos[b][1]-ay))
            ex, ey = env.pos[target]
            dx, dy = ex - ax, ey - ay
            heal_range = UNIT_TYPES[utype]["range"]
            if max(abs(dx), abs(dy)) <= heal_range:
                if abs(dx) >= abs(dy):
                    return 7 if dx < 0 else 8
                else:
                    return 5 if dy < 0 else 6
            else:
                if abs(dx) > abs(dy):
                    return 3 if dx < 0 else 4
                else:
                    return 1 if dy < 0 else 2
        else:
            return 0

    # 攻击者逻辑
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

    if max(abs(dx), abs(dy)) <= attack_range:
        if abs(dx) >= abs(dy):
            return 7 if dx < 0 else 8
        else:
            return 5 if dy < 0 else 6
    else:
        if abs(dx) > abs(dy):
            return 3 if dx < 0 else 4
        else:
            return 1 if dy < 0 else 2


def render_frame(env, cell_size=25):
    return env.render_rgb(cell_size)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--outfile", type=str, default="fight_v6.gif")
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=3.0, help="Frames per second for output GIF")
    args = parser.parse_args()

    env = make_env(grid_size=args.grid_size, seed=args.seed)
    obs, info = env.reset()
    frames = []

    for step in range(args.steps):
        actions = {a: rule_based_policy(env, a) for a in env.agents if env.alive.get(a, False)}
        obs, rew, term, trunc, info = env.step(actions)
        frames.append(render_frame(env))
        if all(term.values()):
            break

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

    frame_duration = 1.0 / args.fps
    imageio.mimsave(args.outfile, frames, duration=frame_duration)
    print(f"🎞️  Animation speed: {args.fps:.1f} FPS ({frame_duration:.2f}s per frame)")
    print(f"✅ Replay saved to {args.outfile}")


if __name__ == "__main__":
    main()
