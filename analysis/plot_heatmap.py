\
import numpy as np
import matplotlib.pyplot as plt
from envs.micro import env as make_env

def collect_positions(episodes=10, grid=15, units=5, seed=0):
    rng = np.random.RandomState(seed)
    heat = np.zeros((grid, grid), dtype=np.float32)
    for ep in range(episodes):
        env = make_env(grid_size=grid, n_per_team=units, seed=rng.randint(0, 100000))
        obs, infos = env.reset()
        for t in range(150):
            positions = [env.pos[a] for a in env.agents if env.alive[a]]
            for (x,y) in positions:
                heat[y,x] += 1
            actions = {a: rng.randint(0,9) if env.alive[a] else 0 for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            if all(terms.values()):
                break
    return heat

if __name__ == "__main__":
    heat = collect_positions(episodes=20, grid=15, units=5, seed=0)
    plt.figure()
    plt.title("Position Heatmap (random policy)")
    plt.imshow(heat, origin="upper")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("v1_heatmap.png", dpi=160)
    print("Saved v1_heatmap.png")
