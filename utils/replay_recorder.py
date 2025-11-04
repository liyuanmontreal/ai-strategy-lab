\
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import imageio
import numpy as np
from envs.micro import env as make_env

def render_frame(env, scale=20):
    # create RGB image from grid render string 
    s = env.render("ansi")
    lines = s.split("\n")
    H = len(lines)
    W = len(lines[0])
    img = np.ones((H*scale, W*scale, 3), dtype=np.uint8) * 255
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "R":
                color = (220, 50, 50)
            elif ch == "B":
                color = (50, 80, 220)
            else:
                color = (230, 230, 230)
            img[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = color
    return img

def rule_based_action(env, rng):
    actions = {}
    positions = {a: env.pos[a] for a in env.agents if env.alive[a]}
    teams = {a: a.split("_")[0] for a in env.agents}
    for a in env.agents:
        if not env.alive[a]:
            actions[a] = 0
            continue
        # simple greedy policy toward nearest enemy; attack if adjacent
        my = positions[a]
        enemies = [b for b in env.agents if teams[b] != teams[a] and b in positions]
        if len(enemies) == 0:
            actions[a] = 0
            continue
        dists = [abs(positions[b][0]-my[0]) + abs(positions[b][1]-my[1]) for b in enemies]
        target = enemies[int(np.argmin(dists))]
        tx, ty = positions[target]
        if (abs(tx - my[0]) + abs(ty - my[1])) == 1:
            if tx == my[0] and ty == my[1]-1: actions[a]=5
            elif tx == my[0] and ty == my[1]+1: actions[a]=6
            elif ty == my[1] and tx == my[0]-1: actions[a]=7
            elif ty == my[1] and tx == my[0]+1: actions[a]=8
        else:
            dx = np.sign(tx - my[0]); dy = np.sign(ty - my[1])
            if abs(tx - my[0]) > abs(ty - my[1]):
                actions[a] = 4 if dx>0 else 3
            else:
                actions[a] = 2 if dy>0 else 1
    return actions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=15)
    parser.add_argument("--units", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--outfile", type=str, default="v1_replay.gif")
    parser.add_argument("--policy", type=str, default="rule_based", choices=["rule_based","random"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    env = make_env(grid_size=args.grid, n_per_team=args.units, seed=args.seed)
    obs, infos = env.reset()

    frames = []
    for t in range(args.steps):
        frames.append(render_frame(env))
        if args.policy == "random":
            actions = {a: rng.randint(0,9) if env.alive[a] else 0 for a in env.agents}
        else:
            actions = rule_based_action(env, rng)
        obs, rewards, terms, truncs, infos = env.step(actions)
        if all(terms.values()):
            frames.append(render_frame(env))
            break

    imageio.mimsave(args.outfile, frames, duration=0.08)
    print("Saved", args.outfile)

if __name__ == "__main__":
    main()
