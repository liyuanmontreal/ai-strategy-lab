\
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from envs.micro import env as make_env
from time import sleep

# Simple heuristic:
# - If enemy is adjacent in a cardinal direction, attack toward it.
# - Else, move toward the nearest enemy (greedy Manhattan).
# - 10% randomization to avoid deadlocks.

def choose_action(agent_name, observation, agents, positions, teams, rng, epsilon=0.1):
    if rng.rand() < epsilon:
        return rng.randint(0, 9)

    # Decode obs for simplicity: we already have positions map passed in
    me = agent_name
    my_pos = positions[me]
    my_team = teams[me]

    # find nearest enemy
    enemies = [a for a in agents if teams[a] != my_team and a in positions]
    if not enemies:
        return 0  # stay

    dists = [abs(positions[a][0]-my_pos[0]) + abs(positions[a][1]-my_pos[1]) for a in enemies]
    target = enemies[int(np.argmin(dists))]
    tx, ty = positions[target]
    dx = np.sign(tx - my_pos[0])
    dy = np.sign(ty - my_pos[1])

    # If adjacent in cardinal direction -> attack
    if (abs(tx - my_pos[0]) + abs(ty - my_pos[1])) == 1:
        if tx == my_pos[0] and ty == my_pos[1]-1:
            return 5  # atk up
        if tx == my_pos[0] and ty == my_pos[1]+1:
            return 6  # atk down
        if ty == my_pos[1] and tx == my_pos[0]-1:
            return 7  # atk left
        if ty == my_pos[1] and tx == my_pos[0]+1:
            return 8  # atk right

    # else move greedily
    if abs(tx - my_pos[0]) > abs(ty - my_pos[1]):
        # move horizontally
        return 4 if dx > 0 else 3
    else:
        return 2 if dy > 0 else 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=15)
    parser.add_argument("--units", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    for ep in range(args.episodes):
        env = make_env(grid_size=args.grid, n_per_team=args.units, seed=rng.randint(0, 100000))
        obs, infos = env.reset()
        done = False
        ep_ret = {a: 0.0 for a in env.agents}
        while True:
            # Build helper dicts
            positions = {a: env.pos[a] for a in env.agents if env.alive[a]}
            teams = {a: a.split("_")[0] for a in env.agents}

            actions = {}
            for a in env.agents:
                if env.alive[a]:
                    actions[a] = choose_action(a, obs[a], env.agents, positions, teams, rng)
                else:
                    actions[a] = 0

            obs, rewards, terms, truncs, infos = env.step(actions)
            for k, v in rewards.items():
                ep_ret[k] += v
            if args.render:
                print(env.render())
                sleep(0.05)

            if all(terms.values()):
                break

        red = [k for k in ep_ret if k.startswith("red")]
        blue = [k for k in ep_ret if k.startswith("blue")]
        red_ret = sum(ep_ret[k] for k in red)
        blue_ret = sum(ep_ret[k] for k in blue)
        print(f"[Episode {ep}] return: red={red_ret:.2f}, blue={blue_ret:.2f}")
        env.close()

if __name__ == "__main__":
    main()
