"""
Stove + Cloth metrics (no robot, no damage env)
- Environment copied from cloth_showcase.py (stove + dishtowel cloth)
- Tracks ONLY:
  * stove temperature
  * distance from cloth centroid to burner (top center of stove AABB)
- Also integrates a simple cloth temperature update like OG's HeatSourceOrSink->Temperature: T += (T_src - T)*rate*dt when within threshold
- Saves a short MP4 video with overlays
"""

import os
import cv2
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.constants import PrimType

# Enable object states and GPU dynamics for cloth
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False


def main():
    """
    Tracks and overlays stove temperature, cloth-to-burner distance, and a simple cloth temperature in a stove+cloth scene.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Scene and objects (copied from cloth_showcase.py)
    scene_cfg = {"type": "Scene"}

    cloth_cfg = {
        "type": "DatasetObject",
        "name": "dishtowel",
        "category": "dishtowel",
        "model": "dtfspn",
        "prim_type": PrimType.CLOTH,
        "abilities": {"cloth": {}},
        # "position": [0.0, -0.9, 0.9],
        "position": [0.2, -1.1, 0.8],
    }

    stove_cfg = {
        "type": "DatasetObject",
        "name": "stove",
        "category": "stove",
        "model": "yhjzwg",
        "position": [0.0, -1.0, 0.0],
        "bounding_box": [0.8, 0.66, 0.65],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "abilities": {
            "heatSource": {
                "temperature": 180.0,
                "heating_rate": 0.15,
                "distance_threshold": 0.25,
                "requires_toggled_on": False,
            },
        },
    }

    cfg = {"scene": scene_cfg, "objects": [stove_cfg, cloth_cfg]}

    # Create environment
    env = og.Environment(configs=cfg)

    # Get references
    cloth = env.scene.object_registry("name", "dishtowel")
    stove = env.scene.object_registry("name", "stove")

    # Camera view
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.6, -2.7, 1.2]),
        orientation=th.tensor([0.59, -0.01, -0.02, 0.81]),
    )

    # Burner point: top center of stove AABB
    stove_center = stove.aabb_center
    stove_extent = stove.aabb_extent
    burner_point = th.tensor([
        stove_center[0].item(),
        stove_center[1].item(),
        (stove_center[2] + 0.5 * stove_extent[2]).item(),
    ], dtype=th.float32)

    # Move cloth right above the burner to start
    try:
        start_above = burner_point + th.tensor([0.0, 0.0, 0.25])
        cloth.set_position_orientation(position=start_above, orientation=th.tensor([0.0, 0.0, 0.0, 1.0]))
    except Exception:
        pass

    # Let cloth settle a bit
    for _ in range(30):
        og.sim.step()

    # Heating parameters (mirror OG HeatSourceOrSink update)
    HEATING_RATE = 0.15
    DIST_THRESH = 0.25
    DEFAULT_STOVE_TEMP = 180.0
    cloth_temp = 20.0  # start temp (C)

    # Track metrics and frames
    images = []
    stove_temps = []
    cloth_distances = []
    cloth_temps = []

    max_steps = 600
    for _ in range(max_steps):
        env.step(th.empty(0))

        # Stove temperature
        try:
            from omnigibson import object_states
            temp = stove.states[object_states.Temperature].get_value()
        except Exception:
            temp = DEFAULT_STOVE_TEMP
        stove_temps.append(float(temp))

        # Cloth centroid distance to burner
        try:
            positions = cloth.root_link.compute_particle_positions()
            centroid = positions.mean(dim=0)
            distance = th.norm(centroid - burner_point).item()
        except Exception:
            distance = 0.0
        cloth_distances.append(distance)

        # Simple cloth temperature update: T += (T_src - T) * rate * dt if within threshold
        dt = og.sim.get_sim_step_dt()
        if distance <= DIST_THRESH:
            cloth_temp = cloth_temp + (float(temp) - cloth_temp) * HEATING_RATE * dt
        cloth_temps.append(cloth_temp)

        # Frame capture + overlay (metrics)
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb = rgb.cpu().numpy()[:, :, :3]
        rgb = cv2.resize(rgb, (640, 480))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        y = 28
        cv2.putText(frame, f"Stove Temp (C): {temp:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y += 28
        cv2.putText(frame, f"Cloth->Burner Distance (m): {distance:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y += 28
        cv2.putText(frame, f"Cloth Temp (C): {cloth_temp:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        images.append(np.ascontiguousarray(frame, dtype=np.uint8))

    # Save video
    os.makedirs('videos_and_images', exist_ok=True)
    avi_path = 'videos_and_images/cloth_stove_metrics.avi'
    mp4_path = 'videos_and_images/cloth_stove_metrics.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(avi_path, fourcc, 30, (images[0].shape[1], images[0].shape[0]))
    for img in images:
        out.write(img)
    out.release()

    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', avi_path, '-c:v', 'mpeg4', mp4_path], check=True)
    os.remove(avi_path)

    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main() 