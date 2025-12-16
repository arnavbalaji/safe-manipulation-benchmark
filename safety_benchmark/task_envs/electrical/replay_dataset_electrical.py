import os
import sys
from typing import Optional

import numpy as np
import cv2

import omnigibson as og
from omnigibson.macros import gm

import h5py


def _ensure_omnigibson_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    og_src = os.path.join(repo_root, "OmniGibson")
    if og_src not in sys.path:
        sys.path.insert(0, og_src)


_ensure_omnigibson_on_path()

from safety_benchmark.damageable_env import DamageableEnvironment  # noqa: E402
from safety_benchmark.task_envs.electrical.collect_dataset_electrical import (  # noqa: E402
    make_base_env,
    reset_episode,
    env_step_with_full_action,
)


# ---------------------------------------------------------------------------
# User-editable config (no CLI)
# ---------------------------------------------------------------------------

# Path to the offline HDF5 dataset to replay
DATASET_PATH = (
    "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/"
    "electrical_teleop_demos.hdf5"
)

# Scene JSON to load (should match training / collection)
SCENE_FILE = (
    "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/reset_saved.json"
)

# How many demos to replay (None = all)
NUM_DEMOS_TO_REPLAY: Optional[int] = 2

# Maximum episode steps during replay (used only as a safety cap; dataset itself
# defines its own episode lengths)
MAX_EPISODE_STEPS = 300

# Directory to save replay videos (status-border sims for first two demos)
REPLAY_VIDEOS_DIR = (
    "safe-manipulation-benchmark/safety_benchmark/task_envs/electrical/replay_videos"
)
os.makedirs(REPLAY_VIDEOS_DIR, exist_ok=True)


def _load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    demos = []
    with h5py.File(path, "r") as f:
        demo_names = sorted([k for k in f.keys() if k.startswith("demo_")])
        if NUM_DEMOS_TO_REPLAY is not None:
            demo_names = demo_names[: NUM_DEMOS_TO_REPLAY]

        if not demo_names:
            raise RuntimeError(f"No 'demo_XXX' groups found in {path}")

        for name in demo_names:
            g = f[name]
            obs = g["observations"][...].astype(np.float32)  # (T+1, obs_dim)
            actions = g["actions"][...].astype(np.float32)  # (T, act_dim)
            dist_rewards = g["distance_reward_terms"][...].astype(np.float32)
            dmg_rewards = g["damage_reward_terms"][...].astype(np.float32)
            dist_terms = g["distance_termination_terms"][...].astype(np.uint8)
            dmg_terms = g["damage_termination_terms"][...].astype(np.uint8)

            demos.append(
                {
                    "name": name,
                    "observations": obs,
                    "actions": actions,
                    "distance_rewards": dist_rewards,
                    "damage_rewards": dmg_rewards,
                    "distance_terminated": dist_terms,
                    "damage_terminated": dmg_terms,
                }
            )
    return demos


def _agent_to_full_action(state: dict, agent_action: np.ndarray) -> np.ndarray:
    """Map PPO agent action ([-1,1]^act_dim) to full robot action (action_dim),
    using the same mapping as OGSimpleEnv.step in the training script.
    """
    action_dim = state["action_dim"]
    right_arm_idx = state["right_arm_idx"]
    right_grip_idx = state["right_grip_idx"]
    arm_input_dim = state["arm_input_dim"]
    arm_action_scale = state["arm_action_scale"]

    a = np.asarray(agent_action, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)
    full = np.zeros((action_dim,), dtype=np.float32)
    arm_action_full = np.zeros((arm_input_dim,), dtype=np.float32)
    arm_action_full[: a.shape[0]] = a[:arm_input_dim]
    full[right_arm_idx] = arm_action_full * arm_action_scale
    full[right_grip_idx] = 1.0
    return full


def main():
    print(f"[Replay] Loading dataset from: {os.path.abspath(DATASET_PATH)}")
    demos = _load_dataset(DATASET_PATH)
    print(f"[Replay] Found {len(demos)} demos to replay.")

    # Build base environment (same as in collector)
    state = make_base_env(scene_file=SCENE_FILE, max_episode_steps=MAX_EPISODE_STEPS)
    env: DamageableEnvironment = state["env"]

    for demo_idx, demo in enumerate(demos):
        name = demo["name"]
        actions = demo["actions"]
        dist_rewards = demo["distance_rewards"]
        dmg_rewards = demo["damage_rewards"]
        dist_terms = demo["distance_terminated"]
        dmg_terms = demo["damage_terminated"]

        print(f"\n[Replay] === Demo {demo_idx + 1}/{len(demos)}: {name} ===")
        print(
            f"[Replay] Steps: {actions.shape[0]}, "
            f"sum(distance_reward)={dist_rewards.sum():.3f}, "
            f"sum(damage_reward)={dmg_rewards.sum():.3f}"
        )

        # Reset env as in data collection
        _ = reset_episode(state)

        # Get laptop object once per demo for damage-status tracking
        laptop = None
        try:
            laptop = env.scene.object_registry("name", "laptop")
        except Exception:
            laptop = None

        # For the first two demos, record a sim video with damage-status border
        record_status_border = demo_idx < 2
        border_frames = [] if record_status_border else None
        damage_status_series = [] if record_status_border else None
        fps = 30

        total_dist_r = 0.0
        total_dmg_r = 0.0
        step_idx = 0

        for t in range(actions.shape[0]):
            agent_action = actions[t]
            full_action = _agent_to_full_action(state, agent_action)

            # Step using the same helper as in collection (adds water, keeps laptop pose, etc.)
            obs_raw, _reward_env, _terminated_env, _truncated_env, _info_env = env_step_with_full_action(
                state, full_action
            )

            # Capture sim frame and laptop damage status for status-border video
            if record_status_border and border_frames is not None and damage_status_series is not None:
                try:
                    rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
                    rgb_np = rgb.cpu().numpy()[:, :, :3]
                    frame = cv2.resize(rgb_np, (1080, 720))
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    border_frames.append(np.ascontiguousarray(frame_bgr.copy(), dtype=np.uint8))
                    if laptop is not None and hasattr(laptop, "damage_status"):
                        status = str(laptop.damage_status).lower()
                    else:
                        status = "none"
                    damage_status_series.append(status)
                except Exception:
                    # If camera or sim not ready, skip this frame
                    pass

            total_dist_r += float(dist_rewards[t])
            total_dmg_r += float(dmg_rewards[t])
            step_idx += 1

            # Respect dataset terminations for debugging
            distance_done = bool(dist_terms[t])
            damage_done = bool(dmg_terms[t])
            if distance_done or damage_done or step_idx >= MAX_EPISODE_STEPS:
                print(
                    f"[Replay] Stopped at step {t + 1}/{actions.shape[0]} "
                    f"(distance_terminated={distance_done}, damage_terminated={damage_done})."
                )
                break

        print(
            f"[Replay] Accumulated distance_reward={total_dist_r:.3f}, "
            f"damage_reward={total_dmg_r:.3f}"
        )

        # Save status-border sim video for first two demos
        if record_status_border and border_frames is not None and len(border_frames) > 0:
            video_name = f"electrical_replay_demo_{demo_idx + 1}_status_border.mp4"
            video_path = os.path.join(REPLAY_VIDEOS_DIR, video_name)
            h, w = border_frames[0].shape[:2]
            border_width = 30
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw_border = cv2.VideoWriter(
                video_path,
                fourcc,
                fps,
                (w + 2 * border_width, h + 2 * border_width),
            )

            # Pad statuses if necessary
            if len(damage_status_series) < len(border_frames):
                damage_status_series.extend(
                    ["none"] * (len(border_frames) - len(damage_status_series))
                )

            for frame_bgr, status in zip(border_frames, damage_status_series):
                s = (status or "none").lower()
                if s in ("major", "critical"):
                    bgr = (0, 0, 255)  # Red
                elif s == "minor":
                    bgr = (0, 255, 255)  # Yellow
                else:
                    bgr = (0, 255, 0)  # Green
                bordered = cv2.copyMakeBorder(
                    frame_bgr,
                    border_width,
                    border_width,
                    border_width,
                    border_width,
                    cv2.BORDER_CONSTANT,
                    value=bgr,
                )
                vw_border.write(np.ascontiguousarray(bordered, dtype=np.uint8))

            vw_border.release()
            print(f"[Replay] Saved status-border sim video: {os.path.abspath(video_path)}")

    print("[Replay] Done replaying all selected demos.")

    try:
        og.clear()
        og.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()

