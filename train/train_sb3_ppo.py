import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from pettingzoo.utils import parallel_to_aec
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from envs.micro import env as make_env

class PrintCallback(BaseCallback):
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.verbose:
                print(f"[SB3] steps={self.n_calls}")
        return True

def build_vec_env(grid_size=12, units=4, seed=42, num_envs=8):
    pz_parallel = make_env(grid_size=grid_size, n_per_team=units, seed=seed)
    vec = ss.pettingzoo_env_to_vec_env_v1(pz_parallel)
    vec = ss.concat_vec_envs_v1(vec, num_envs, num_cpus=1, base_class="stable_baselines3")
    return vec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--save-path", type=str, default="checkpoints/sb3_ppo_v1")
    args = parser.parse_args()

    vec_env = build_vec_env(grid_size=args.grid_size, units=args.units, seed=args.seed, num_envs=args.num_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.lr,
        n_steps=max(64, args.batch_size // max(1, args.num_envs)),
        batch_size=256,
        ent_coef=args.ent_coef,
        gamma=0.97,
        verbose=1,
        #seed=args.seed,
        device="cpu"
    )

    cb = PrintCallback(check_freq=10000, verbose=1)
    model.learn(total_timesteps=args.total_steps, callback=cb)
    model.save(args.save_path)
    print("Saved policy to", args.save_path)

if __name__ == "__main__":
    main()
