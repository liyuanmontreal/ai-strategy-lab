\
import argparse
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.qmix.qmix import QMIXConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.utils import parallel_to_aec
from envs.micro import env as make_env

# Note: RLlib expects a PettingZoo AEC env wrapped by PettingZooEnv.
# Our env is parallel, so we convert with parallel_to_aec.

def env_creator(config):
    grid = config.get("grid_size", 12)
    units = config.get("n_per_team", 4)
    seed = config.get("seed", 42)
    return parallel_to_aec(make_env(grid_size=grid, n_per_team=units, seed=seed))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=100_000)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Register env via tune (lambda recommended in new RLlib)
    def mk():
        return env_creator({"grid_size": args.grid_size, "n_per_team": args.units, "seed": args.seed})
    rllib_env = PettingZooEnv(mk())

    # Homogeneous agents: share groups across teams
    obs_space = rllib_env.observation_space
    act_space = rllib_env.action_space

    config = (
        QMIXConfig()
        .environment(env=rllib_env, disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=args.num_workers)
        .training(
            gamma=0.97,
            lr=1e-3,
            train_batch_size=4000,
            n_step=2,
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *a, **k: "shared_policy",
            policies_to_train=["shared_policy"],
        )
        .resources(num_gpus=0)
    )

    algo = config.build()

    results = None
    total_ts = 0
    while total_ts < args.train_steps:
        result = algo.train()
        total_ts = result["timesteps_total"]
        print(f"Iter={result['training_iteration']} ts={total_ts} "
              f"episode_reward_mean={result.get('episode_reward_mean')}")
        results = result

    # Save checkpoint
    ckpt = algo.save(checkpoint_dir="./checkpoints")
    print("Saved checkpoint to", ckpt)

    ray.shutdown()

if __name__ == "__main__":
    main()
