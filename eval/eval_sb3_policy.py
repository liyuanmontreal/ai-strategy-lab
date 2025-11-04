import argparse
import numpy as np
from envs.micro_v1 import env as make_env
from pettingzoo.utils import parallel_to_aec
from stable_baselines3 import PPO

def rule_based_action(env, rng):
    actions = {}
    positions = {a: env.pos[a] for a in env.agents if env.alive[a]}
    teams = {a: a.split("_")[0] for a in env.agents}
    for a in env.agents:
        if not env.alive[a]:
            actions[a] = 0
            continue
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
    parser.add_argument("--model", type=str, default="checkpoints/sb3_ppo_v1.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    model = PPO.load(args.model, device="cpu")

    red_wins = 0
    blue_wins = 0
    for ep in range(args.episodes):
        p_env = make_env(grid_size=args.grid_size, n_per_team=args.units, seed=rng.randint(0, 10**6))
        env = parallel_to_aec(p_env)
        obs, infos = env.reset()
        while True:
            actions = {}
            # SB3 controls red; blue is rule-based
            rb_actions = rule_based_action(env, rng)
            for a in env.agents:
                if not env.alive[a]:
                    actions[a] = 0
                    continue
                if a.startswith("red"):
                    action, _ = model.predict(obs[a], deterministic=True)
                    actions[a] = int(action)
                else:
                    actions[a] = rb_actions.get(a, 0)
            obs, rewards, terms, truncs, infos = env.step(actions)
            if all(terms.values()):
                break

        red_alive = any(env.alive.get(a, False) for a in env.agents if a.startswith("red"))
        blue_alive = any(env.alive.get(a, False) for a in env.agents if a.startswith("blue"))
        if red_alive and not blue_alive:
            red_wins += 1
        elif blue_alive and not red_alive:
            blue_wins += 1

        env.close()

    print(f"Eval vs Rule-based: RED(SB3) wins={red_wins}/{args.episodes}, BLUE(RB) wins={blue_wins}/{args.episodes}")

if __name__ == "__main__":
    main()
