import os
import time
import random

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import omnigibson as og

# Reuse environment, agent, BC utilities, and eval helpers from the offline training script
from safety_benchmark.task_envs.electrical.ppo_train_cleanrl_electrical_offline import (
    Args,
    make_env,
    Agent,
    run_eval,
    DEFAULT_SCENE_FILE,
    _load_bc_dataset_from_hdf5,
    BC_INIT_CHECKPOINT,
    STATUS_BORDER_EPISODES,
)

# Also import the module itself so we can override STATUS_BORDER_EPISODES for BC eval
import safety_benchmark.task_envs.electrical.ppo_train_cleanrl_electrical_offline as electrical_offline


EVAL_VIDEOS_DIR = (
    "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/eval_videos_bc"
)
os.makedirs(EVAL_VIDEOS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Simple configuration knobs (edit these variables directly)
# ----------------------------------------------------------------------
SEED = 1
CAPTURE_VIDEO = False
BC_USE_ONLY_SAFE_DEMOS = True  # True => only demos with total_damage == 0.0
BC_NUM_EPOCHS = 20
NUM_EVAL_EPISODES = 10
# Optional: path to a saved BC policy checkpoint to load and evaluate.
# If set to None, this script will train the BC policy from scratch using the
# offline dataset before evaluation.
# BC_POLICY_CHECKPOINT = "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/checkpoints/checkpoint_all_demos.pth"
BC_POLICY_CHECKPOINT = None


def run_offline_bc_pretrain_eval_only(args: Args, agent: Agent, optimizer: optim.Optimizer, device: torch.device):
    """
    Eval-only variant of BC pretraining that does NOT use wandb.
    Trains only the actor via behavior cloning on the offline dataset.
    """
    obs_arr, act_arr = _load_bc_dataset_from_hdf5(
        args.offline_dataset_path, args.bc_use_only_safe_demos
    )
    if obs_arr is None:
        print("[BC Eval] No BC data available; skipping BC pretraining.")
        return

    obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=device)
    act_tensor = torch.tensor(act_arr, dtype=torch.float32, device=device)
    dataset_size = obs_tensor.shape[0]

    print(
        f"[BC Eval] Starting behavior cloning pretraining on {dataset_size} transitions, "
        f"epochs={args.bc_num_epochs}, safe_only={args.bc_use_only_safe_demos}."
    )

    for bc_epoch in range(1, args.bc_num_epochs + 1):
        inds = np.arange(dataset_size)
        np.random.shuffle(inds)

        epoch_losses = []
        step = max(1, int(args.minibatch_size) or 1)
        for start in range(0, dataset_size, step):
            end = start + step
            mb_inds = inds[start:end]
            if mb_inds.size == 0:
                continue
            mb_obs = obs_tensor[mb_inds]
            mb_actions = act_tensor[mb_inds]

            _, logprob, _, _ = agent.get_action_and_value(mb_obs, action=mb_actions)
            bc_loss = -logprob.mean()

            optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_losses.append(float(bc_loss.item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(
            f"[BC Eval] Epoch {bc_epoch}/{args.bc_num_epochs} "
            f"BC loss={mean_loss:.6f}"
        )


def parse_cli_overrides(args: Args):
    """
    Minimal CLI parser to control BC pretraining behavior for evaluation.

    We intentionally keep this light and only expose what's needed:
    - seed
    - whether to capture videos
    - whether to use only safe demos for BC
    - optional override for number of BC epochs
    """
    parser = argparse.ArgumentParser(description="BC-only evaluation on electrical task")
    parser.add_argument(
        "--seed",
        type=int,
        default=args.seed,
        help="Random seed (overrides Args.seed)",
    )
    parser.add_argument(
        "--capture-video",
        action="store_true",
        help="If set, capture evaluation videos (overrides Args.capture_video=True)",
    )
    parser.add_argument(
        "--bc-safe-only",
        action="store_true",
        help="If set, use only safe demos (total_damage == 0.0) for BC pretraining.",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=20,
        help="Number of BC epochs to run before evaluation (default: 20).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to run (default: 10).",
    )

    cli_args = parser.parse_args()

    # Apply overrides into Args
    args.seed = int(cli_args.seed)
    if cli_args.capture_video:
        args.capture_video = True
    # Keep BC pretraining enabled and configure its details
    args.bc_pretrain_enabled = True
    args.bc_use_only_safe_demos = bool(cli_args.bc_safe_only)
    args.bc_num_epochs = int(cli_args.bc_epochs)

    return args, cli_args.num_episodes


def main():
    # Start from the offline Args defaults (includes offline dataset path + BC knobs)
    args = Args()

    # Force single-env evaluation for OG (same as ppo_eval)
    if args.num_envs != 1:
        print(f"[BC Eval] num_envs={args.num_envs} not supported; forcing num_envs=1.")
        args.num_envs = 1

    # Apply config overrides (seed, capture_video, BC options)
    args.seed = int(SEED)
    args.capture_video = bool(CAPTURE_VIDEO)
    args.bc_pretrain_enabled = True
    args.bc_use_only_safe_demos = bool(BC_USE_ONLY_SAFE_DEMOS)
    args.bc_num_epochs = int(BC_NUM_EPOCHS)
    num_eval_episodes = int(NUM_EVAL_EPISODES)

    # Ensure batch_size and minibatch_size are set as in offline training,
    # so that BC pretraining has a valid minibatch_size (>0).
    # From ppo_train_cleanrl_electrical_offline.py:
    #   args.batch_size = int(args.num_envs * args.num_steps)
    #   args.minibatch_size = int(args.batch_size // args.num_minibatches)
    #   args.num_iterations = args.total_timesteps // args.batch_size
    args.batch_size = int(args.num_envs * args.num_steps)
    if args.num_minibatches > 0:
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
    else:
        args.minibatch_size = args.batch_size
    if args.batch_size > 0:
        args.num_iterations = args.total_timesteps // args.batch_size

    # Seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Build env exactly as in offline training
    scene_file = DEFAULT_SCENE_FILE
    run_name = (
        f"OG-Scene__electrical_bc_eval__{args.seed}__{int(time.time())}__{random.randint(1000, 9999)}"
    )
    envs = gym.vector.SyncVectorEnv(
        [make_env(scene_file, 0, args.capture_video, run_name)]
    )

    # Base env for direct access (same pattern as training script)
    base_env = envs.envs[0]
    while hasattr(base_env, "env"):
        base_env = base_env.env

    # Create agent with the same architecture as in offline training
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ----------------------------------------------------------------------
    # 1) Initialize BC policy: either load from checkpoint or run BC pretraining
    # ----------------------------------------------------------------------
    if BC_POLICY_CHECKPOINT is not None:
        if os.path.exists(BC_POLICY_CHECKPOINT):
            print(
                f"[BC Eval] Loading BC policy checkpoint from: "
                f"{os.path.abspath(BC_POLICY_CHECKPOINT)}"
            )
            ckpt = torch.load(BC_POLICY_CHECKPOINT, map_location=device)
            state_dict = ckpt.get("agent_state_dict", ckpt)
            agent.load_state_dict(state_dict)
            print("[BC Eval] Loaded BC policy from checkpoint. Skipping BC pretraining.\n")
        else:
            print(
                f"[BC Eval] Warning: BC_POLICY_CHECKPOINT '{BC_POLICY_CHECKPOINT}' "
                f"not found. Falling back to BC pretraining from scratch."
            )
            print(
                f"[BC Eval] Running offline behavior cloning pretraining for "
                f"{args.bc_num_epochs} epochs "
                f"(safe_only={args.bc_use_only_safe_demos})..."
            )
            run_offline_bc_pretrain_eval_only(args, agent, optimizer, device)
            print("[BC Eval] BC pretraining complete. Proceeding to evaluation.\n")
    else:
        print(
            f"[BC Eval] No BC_POLICY_CHECKPOINT provided. "
            f"Running BC pretraining from scratch for {args.bc_num_epochs} epochs "
            f"(safe_only={args.bc_use_only_safe_demos})..."
        )
        run_offline_bc_pretrain_eval_only(args, agent, optimizer, device)
        print("[BC Eval] BC pretraining complete. Proceeding to evaluation.\n")

    # ----------------------------------------------------------------------
    # 2) Evaluate the BC-only policy on the environment, using the same
    #    metrics and video pipeline as the standard eval script.
    # ----------------------------------------------------------------------
    print(f"[BC Eval] Running {num_eval_episodes}-episode evaluation of BC policy...")
    # For BC eval, record damage-status-border sim videos for ALL episodes, not just the first two.
    # We do this by overriding the shared STATUS_BORDER_EPISODES constant before calling run_eval.
    electrical_offline.STATUS_BORDER_EPISODES = num_eval_episodes
    (
        success_rate,
        eval_avg_return,
        saved_videos,
        task_success_rate,
        damage_failure_rate,
        avg_steps_to_solve,
        mean_total_damage,
        safe_success_rate,
    ) = run_eval(
        base_env,
        agent,
        device,
        EVAL_VIDEOS_DIR,
        iteration_num=0,
        num_episodes=num_eval_episodes,
    )

    # Print metrics (copying the style of ppo_eval_cleanrl_electrical.py)
    print("\n========== Electrical Task BC-Only Evaluation Summary ==========")
    print(f"BC epochs                 : {args.bc_num_epochs}")
    print(f"BC safe-only demos        : {args.bc_use_only_safe_demos}")
    print(f"Task success rate         : {task_success_rate:.3f}")
    print(f"Damage failure rate       : {damage_failure_rate:.3f}")
    print(f"Safe success rate         : {safe_success_rate:.3f} (success with no damage)")
    print(f"Avg steps to solve        : {avg_steps_to_solve:.1f}")
    print(f"Mean total damage/episode : {mean_total_damage:.1f} (max possible 300.0)")
    print(f"Eval success rate (legacy): {success_rate:.3f}")
    print(f"Eval average return       : {eval_avg_return:.3f}")
    if saved_videos:
        print("Saved evaluation videos   :")
        for p in saved_videos:
            print(f"  - {os.path.abspath(p)}")
    else:
        print("Saved evaluation videos   : (none)")
    print("===============================================================\n")

    # Save BC-init checkpoint for reuse in the offline training script.
    try:
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "bc_epochs": int(args.bc_num_epochs),
                "bc_safe_only": bool(args.bc_use_only_safe_demos),
                "seed": int(args.seed),
            },
            BC_INIT_CHECKPOINT,
        )
        print(f"[BC Eval] Saved BC init checkpoint to: {os.path.abspath(BC_INIT_CHECKPOINT)}")
    except Exception as e:
        print(f"[BC Eval] Warning: Failed to save BC init checkpoint: {e}")

    # Clean up
    envs.close()
    og.shutdown()


if __name__ == "__main__":
    main()

