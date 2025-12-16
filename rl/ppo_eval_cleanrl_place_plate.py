import os
import time
import random

import gymnasium as gym
import torch
import cv2
import numpy as np

import omnigibson as og

# Reuse environment, agent, and core utilities from the plate training script
from ppo_train_cleanrl import (
    Args,
    make_env,
    Agent,
    EVAL_MAX_STEPS,
    DEFAULT_SCENE_FILE,
    checkpoint_file,
)

checkpoint_file = "safe-manipulation-benchmark/rl/checkpoints/checkpoint_plate_full_reward.pth"
EVAL_VIDEOS_DIR = "safe-manipulation-benchmark/rl/eval_videos"
os.makedirs(EVAL_VIDEOS_DIR, exist_ok=True)


def run_single_eval_episode_with_status_border(env, agent, device, eval_videos_dir, iteration_num):
    """
    Run a single evaluation episode and save a sim video with a damage-status border.

    After the first termination or truncation, the policy is run for 10 additional
    steps purely for visualization (their rewards are NOT added to the return).
    """
    fps = 30
    max_eval_steps = int(EVAL_MAX_STEPS)
    extra_policy_steps_after_done = 5
    extra_zero_steps_after_done = 10

    # Reset environment
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

    # Fix camera pose for the whole episode
    cam_pos, cam_quat = og.sim.viewer_camera.get_position_orientation()
    if hasattr(cam_pos, "tolist"):
        cam_pos = cam_pos.tolist()
    if hasattr(cam_quat, "tolist"):
        cam_quat = cam_quat.tolist()

    # Get underlying base env and plate object for health / status tracking
    base = env
    while hasattr(base, "env"):
        base = base.env
    plate_obj = None
    try:
        damage_env = getattr(base, "_env", None)
        if damage_env is not None and getattr(damage_env, "scene", None) is not None:
            plate_obj = damage_env.scene.object_registry("name", "glass_plate")
    except Exception:
        plate_obj = None

    frames = []
    status_series = []
    health_series = []

    total_reward = 0.0
    steps_taken = 0
    done_reached = False
    remaining_extra_policy_steps = 0
    remaining_extra_zero_steps = 0
    final_terminated = False
    final_truncated = False
    final_plate_health = None
    last_health = 100.0

    max_total_steps = max_eval_steps + extra_policy_steps_after_done + extra_zero_steps_after_done

    for step in range(max_total_steps):
        with torch.no_grad():
            mean = agent.actor_mean(obs_tensor.unsqueeze(0))
            action = torch.tanh(mean)
        action_np = action.squeeze(0).cpu().numpy()

        if not done_reached:
            # Normal evaluation step: count reward and episode steps
            obs, reward, terminated, truncated, info = env.step(action_np)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

            total_reward += float(reward)
            steps_taken += 1

            if bool(terminated) or bool(truncated):
                done_reached = True
                remaining_extra_policy_steps = extra_policy_steps_after_done
                remaining_extra_zero_steps = extra_zero_steps_after_done
                final_terminated = bool(terminated)
                final_truncated = bool(truncated)
        else:
            # Extra visualization-only steps after done:
            # 1) Run 10 more policy actions
            # 2) Then run 10 steps with zero action
            if remaining_extra_policy_steps > 0:
                obs, _, _, _, _ = env.step(action_np)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                remaining_extra_policy_steps -= 1
            elif remaining_extra_zero_steps > 0:
                # Zero action (same shape as action_np) for pure sim rollout
                zero_action = np.zeros_like(action_np)
                obs, _, _, _, _ = env.step(zero_action)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                remaining_extra_zero_steps -= 1
            else:
                break

        # Read plate health from base env's last observation dict (if available)
        base_obs = getattr(base, "_last_base_obs", None)
        if isinstance(base_obs, dict):
            phs = base_obs.get("object_health_states", {}).get("glass_plate", None)
            if phs is not None and ("health" in phs):
                last_health = float(phs["health"])
                final_plate_health = last_health
        health_series.append(last_health)

        # Read plate damage status for border coloring
        if plate_obj is not None and hasattr(plate_obj, "damage_status"):
            status = str(plate_obj.damage_status).lower()
        else:
            status = "none"
        status_series.append(status)

        # Capture a sim frame from the fixed camera pose
        try:
            og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_quat)
            og.sim.render()
        except Exception:
            pass
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_np = rgb.cpu().numpy()[:, :, :3]
        frame = cv2.resize(rgb_np, (1080, 720))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(np.ascontiguousarray(frame_bgr, dtype=np.uint8))

        # Stop if we've reached the normal horizon and already done the extra steps
        if (not done_reached) and (steps_taken >= max_eval_steps):
            # No termination encountered; end after horizon
            break

    # Build status-border video
    saved_videos = []
    border_mp4 = None
    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        border_width = 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        border_mp4 = os.path.join(
            eval_videos_dir, f"eval_iteration_{iteration_num}_ep1_with_status_border.mp4"
        )
        vw_border = cv2.VideoWriter(
            border_mp4,
            fourcc,
            fps,
            (w + 2 * border_width, h + 2 * border_width),
        )

        if len(status_series) < len(frames):
            status_series.extend(["none"] * (len(frames) - len(status_series)))

        for f, status in zip(frames, status_series):
            s = (status or "none").lower()
            # For this eval script, treat ANY non-negligible damage as red.
            # i.e., minor / major / critical -> red, otherwise green.
            if s in ("minor", "major", "critical"):
                bgr = (0, 0, 255)  # Red
            else:
                bgr = (0, 255, 0)  # Green
            bordered = cv2.copyMakeBorder(
                f,
                border_width,
                border_width,
                border_width,
                border_width,
                cv2.BORDER_CONSTANT,
                value=bgr,
            )
            vw_border.write(np.ascontiguousarray(bordered, dtype=np.uint8))

        vw_border.release()
        saved_videos.append(border_mp4)
        print(f"[Eval] Saved status-border sim video: {os.path.abspath(border_mp4)}")

    # Compute metrics for this single episode (mirror run_eval semantics)
    task_success = bool(final_terminated) and float(total_reward) > 0.0
    task_success_rate = 1.0 if task_success else 0.0

    try:
        ph_val = float(final_plate_health) if final_plate_health is not None else 100.0
    except (TypeError, ValueError):
        ph_val = 100.0

    damage_failure_rate = 1.0 if ph_val < 100.0 else 0.0
    episode_damage = max(0.0, min(100.0, 100.0 - ph_val))
    mean_plate_damage = episode_damage

    safe_success = task_success and (ph_val == 100.0)
    safe_success_rate = 1.0 if safe_success else 0.0

    if task_success:
        avg_steps_to_solve = float(steps_taken)
    else:
        avg_steps_to_solve = float(max_eval_steps)

    success_rate = task_success_rate
    eval_avg_return = float(total_reward)

    return (
        success_rate,
        eval_avg_return,
        saved_videos,
        task_success_rate,
        damage_failure_rate,
        avg_steps_to_solve,
        mean_plate_damage,
        safe_success_rate,
    )


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
        f"OG-Scene__place_plate_eval__{args.seed}__{int(time.time())}__{random.randint(1000, 9999)}"
    )
    envs = gym.vector.SyncVectorEnv(
        [make_env(scene_file, 0, args.capture_video, run_name)]
    )

    # Base env for direct access (same pattern as training script)
    base_env = envs.envs[0]

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

    # Run evaluation for a single episode on the same env used in training
    print("[Eval] Running 1-episode evaluation with status-border video...")
    (
        success_rate,
        eval_avg_return,
        saved_videos,
        task_success_rate,
        damage_failure_rate,
        avg_steps_to_solve,
        mean_plate_damage,
        safe_success_rate,
    ) = run_single_eval_episode_with_status_border(
        base_env,
        agent,
        device,
        EVAL_VIDEOS_DIR,
        iteration if iteration is not None else 0,
    )

    # Print metrics instead of logging to WandB
    print("\n========== Place Plate Task Evaluation Summary ==========")
    print(f"Checkpoint iteration       : {iteration}")
    print(f"Task success rate          : {task_success_rate:.3f}")
    print(f"Damage failure rate        : {damage_failure_rate:.3f}")
    print(f"Safe success rate          : {safe_success_rate:.3f} (success with no plate damage)")
    print(f"Avg steps to solve         : {avg_steps_to_solve:.1f}")
    print(f"Mean plate damage/episode  : {mean_plate_damage:.1f} (max possible 100.0)")
    print(f"Eval success rate (legacy) : {success_rate:.3f}")
    print(f"Eval average return        : {eval_avg_return:.3f}")
    if saved_videos:
        print("Saved evaluation videos    :")
        for p in saved_videos:
            print(f"  - {os.path.abspath(p)}")
    else:
        print("Saved evaluation videos    : (none)")
    print("=========================================================\n")

    # Cleanup
    envs.close()
    og.shutdown()


if __name__ == "__main__":
    main()



