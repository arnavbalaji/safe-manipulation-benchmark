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
    Cloth showcase (no damage env):
    - Spawns a cloth (dishtowel) and a stove
    - Steps physics with GPU cloth dynamics
    - Computes and overlays proxy signals relevant to "damage":
      * droop (centroid sag from initial)
      * average particle speed (vibration proxy)
      * proximity exposure score (distance-based accumulation to hot spot)
    - Saves a short MP4 video
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Scene and objects
    scene_cfg = {"type": "Scene"}

    cloth_cfg = {
        "type": "DatasetObject",
        "name": "dishtowel",
        "category": "dishtowel",
        "model": "dtfspn",
        "prim_type": PrimType.CLOTH,
        "abilities": {"cloth": {}},
        "position": [0.0, -0.9, 0.9],
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

    # Add viewer teleoperation (W/A/S/D lateral, G down)
    from omnigibson.utils.ui_utils import CameraMover
    class CustomCameraMover(CameraMover):
        @property
        def input_to_command(self):
            return {
                lazy.carb.input.KeyboardInput.D: th.tensor([self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.A: th.tensor([-self.delta, 0, 0]),
                lazy.carb.input.KeyboardInput.W: th.tensor([0, 0, -self.delta]),
                lazy.carb.input.KeyboardInput.S: th.tensor([0, 0, self.delta]),
                lazy.carb.input.KeyboardInput.G: th.tensor([0, -self.delta, 0]),
            }

    camera_mover = CustomCameraMover(cam=og.sim.viewer_camera, delta=0.1)
    camera_mover.print_info()

    # Let cloth settle a bit
    for _ in range(30):
        og.sim.step()

    # Initial centroid (for droop baseline)
    pos0 = cloth.root_link.compute_particle_positions()
    init_centroid = pos0.mean(dim=0)

    # Estimate stove hot point near its top surface using AABB
    stove_center = stove.aabb_center
    stove_extent = stove.aabb_extent
    hot_point = th.tensor([
        stove_center[0].item(),
        stove_center[1].item(),
        (stove_center[2] + 0.5 * stove_extent[2]).item(),
    ], dtype=th.float32)

    # Tracking
    images = []
    exposure_score = 0.0
    prev_positions = None
    dt = og.sim.get_physics_dt()

    max_steps = 600
    for _ in range(max_steps):
        env.step(th.empty(0))

        # Particle positions
        positions = cloth.root_link.compute_particle_positions()
        centroid = positions.mean(dim=0)

        # Average particle speed (approx)
        if prev_positions is not None and prev_positions.shape == positions.shape:
            vel = (positions - prev_positions) / max(dt, 1e-6)
            avg_speed = th.norm(vel, dim=1).mean().item()
        else:
            avg_speed = 0.0

        prev_positions = positions.clone()

        # Droop (positive sag from initial)
        droop = max(0.0, (init_centroid[2] - centroid[2]).item())

        # Proximity exposure accumulation (distance to hot point)
        dist = th.norm(centroid - hot_point).item()
        exposure_step = 1.0 / (dist + 1e-3)
        exposure_score += exposure_step * dt

        # Frame capture with overlays
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb = rgb.cpu().numpy()[:, :, :3]
        rgb = cv2.resize(rgb, (640, 480))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        y = 28
        cv2.putText(frame, f"Droop (m): {droop:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y += 28
        cv2.putText(frame, f"Avg particle speed (m/s): {avg_speed:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y += 28
        cv2.putText(frame, f"Exposure score: {exposure_score:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        images.append(np.ascontiguousarray(frame, dtype=np.uint8))

    # Save video
    os.makedirs("videos_and_images", exist_ok=True)
    avi_path = "videos_and_images/cloth_showcase.avi"
    mp4_path = "videos_and_images/cloth_showcase.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(avi_path, fourcc, 30, (images[0].shape[1], images[0].shape[0]))
    for img in images:
        writer.write(img)
    writer.release()

    # Convert to mp4 (if ffmpeg available)
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i", avi_path, "-c:v", "mpeg4", mp4_path], check=True)
        os.remove(avi_path)
    except Exception:
        pass

    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main() 