import os
import time
import random

import gymnasium as gym
import torch

import omnigibson as og

# Reuse environment, agent, and eval utilities from the training script
from safety_benchmark.task_envs.electrical.ppo_train_cleanrl_electrical_offline import (
    Args,
    make_env,
    Agent,
    run_eval,
    DEFAULT_SCENE_FILE,
)


EVAL_VIDEOS_DIR = (
    "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/eval_videos"
)
os.makedirs(EVAL_VIDEOS_DIR, exist_ok=True)
checkpoint_file = "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/checkpoints/checkpoint_bootstrap_full.pth"


def main():
    args = Args()

    # For OG we must use a single env to avoid simulator conflicts
    if args.num_envs != 1:
        print(f"[OG Notice] num_envs={args.num_envs} not supported; forcing num_envs=1.")
        args.num_envs = 1

    # Seeding (local to this eval script)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Build env exactly as in training
    scene_file = DEFAULT_SCENE_FILE
    run_name = (
        f"OG-Scene__electrical_eval__{args.seed}__{int(time.time())}__{random.randint(1000, 9999)}"
    )
    envs = gym.vector.SyncVectorEnv(
        [make_env(scene_file, 0, args.capture_video, run_name)]
    )

    # Base env for direct access (same pattern as training script)
    base_env = envs.envs[0]
    while hasattr(base_env, "env"):
        base_env = base_env.env

    # Create agent with the same architecture as in training
    agent = Agent(envs).to(device)

    # Load checkpoint if available
    iteration = 0
    if os.path.exists(checkpoint_file):
        print(f"[Eval] Loading checkpoint from {checkpoint_file} ...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        iteration = int(checkpoint.get("iteration", 0))
        best_eval_avg_return = checkpoint.get("best_eval_avg_return", None)
        print(
            f"[Eval] Loaded checkpoint from iteration {iteration} "
            f"(best eval avg return: {best_eval_avg_return})"
        )
    else:
        print(
            f"[Eval] WARNING: No checkpoint found at {checkpoint_file}. "
            "Using current (untrained) agent parameters."
        )

    # Run evaluation for 10 episodes on the same env used in training
    print("[Eval] Running 10-episode evaluation...")
    success_rate, eval_avg_return, saved_videos, task_success_rate, damage_failure_rate, avg_steps_to_solve, mean_total_damage, safe_success_rate = run_eval(
        base_env,
        agent,
        device,
        EVAL_VIDEOS_DIR,
        iteration if iteration is not None else 0,
        num_episodes=10,
    )

    # Print metrics instead of logging to WandB
    print("\n========== Electrical Task Evaluation Summary ==========")
    print(f"Checkpoint iteration       : {iteration}")
    print(f"Task success rate          : {task_success_rate:.3f}")
    print(f"Damage failure rate        : {damage_failure_rate:.3f}")
    print(f"Safe success rate          : {safe_success_rate:.3f} (success with no damage)")
    print(f"Avg steps to solve         : {avg_steps_to_solve:.1f}")
    print(f"Mean total damage/episode  : {mean_total_damage:.1f} (max possible 300.0)")
    print(f"Eval success rate (legacy) : {success_rate:.3f}")
    print(f"Eval average return        : {eval_avg_return:.3f}")
    if saved_videos:
        print("Saved evaluation videos    :")
        for p in saved_videos:
            print(f"  - {os.path.abspath(p)}")
    else:
        print("Saved evaluation videos    : (none)")
    print("=======================================================\n")

    # Drop into debugger before shutting down the sim
    breakpoint()

    # Cleanup
    envs.close()
    og.shutdown()


if __name__ == "__main__":
    main()


