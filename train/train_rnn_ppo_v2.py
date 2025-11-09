import argparse
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import supersuit as ss
from sb3_contrib import RecurrentPPO
from envs.micro_v2 import env as make_env

def build_vec_env(num_envs=4, grid_size=15, seed=0):
    # PettingZoo ParallelEnv -> VecEnv (参数共享)
    base_env = make_env(grid_size=grid_size, seed=seed)
    vec = ss.pettingzoo_env_to_vec_env_v1(base_env)
    vec = ss.concat_vec_envs_v1(vec, num_envs, num_cpus=1, base_class="stable_baselines3")
    return vec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="checkpoints/rnn_ppo_v2")
    args = parser.parse_args()

    vec_env = build_vec_env(
        num_envs=args.num_envs,
        grid_size=args.grid_size,
        seed=args.seed,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=1,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.97,
        device="cpu",
    )

    model.learn(total_timesteps=args.total_steps)
    model.save(args.save_path)
    print("Saved:", args.save_path)

if __name__ == "__main__":
    main()
