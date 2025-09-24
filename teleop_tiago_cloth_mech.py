"""
Teleop demo: TIAGo + Coffee Table + Cloth (no damage env)
- Mirrors teleop_tiago_bowl.py scene layout: TIAGo and a coffee table
- Replaces bowl/baseball with a cloth (dishtowel) using GPU cloth dynamics
- Metrics:
  * contact_force_proxy: per-step net impulse magnitude aggregated from rigid bodies (table + robot)
  * cloth_impact_proxy: energy-over-distance deceleration-gated proxy from cloth centroid motion
- Saves a short MP4 video with overlays
"""

# Non-interactive backend for matplotlib if used downstream
import matplotlib
matplotlib.use('Agg')

import torch as th
import os
import cv2
import numpy as np
import subprocess

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.constants import PrimType
from omnigibson.utils.ui_utils import KeyboardRobotController
from omnigibson.object_states import OnTop

# Cloth requires GPU dynamics; disable flatcache
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False


def _aggregate_rigid_contact_impulses(rigid_obj):
    """
    Aggregate a proxy for contact force from a rigid object by summing net impulse magnitudes
    over its links for the current physics step.
    """
    total = 0.0
    try:
        for link_name, link in rigid_obj.links.items():
            try:
                contacts_now = link.contact_list()
            except Exception:
                contacts_now = []
            if contacts_now:
                try:
                    forces = th.tensor([c.impulse.tolist() for c in contacts_now])
                    net_vec = th.sum(forces, dim=0)
                    total += th.norm(net_vec).item()
                except Exception:
                    # Fallback: sum magnitudes if vector sum fails
                    try:
                        mags = [th.norm(th.tensor(c.impulse)).item() for c in contacts_now]
                        total += float(sum(mags))
                    except Exception:
                        pass
    except Exception:
        pass
    return float(total)


def main():
    """
    Teleop TIAGo with a coffee table and a cloth. Track contact forces (rigid) and cloth impact proxy.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Scene and robot
    scene_cfg = {"type": "Scene"}

    robot0_cfg = {
        "type": "Tiago",
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
        "position": [0., 0.65, 0.0],
        "orientation": [0, 0, -1, 1],
        "grasping_mode": "assisted",
    }

    # Coffee table (same model as teleop_tiago_bowl.py)
    coffee_table_cfg = {
        "type": "DatasetObject",
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "aoojzy",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 0.7071068, 0.7071068],
        "scale": [1.0, 1.0, 1.0],
    }

    # Cloth object
    cloth_cfg = {
        "type": "DatasetObject",
        "name": "cloth",
        "category": "dishtowel",
        "model": "dtfspn",
        "prim_type": PrimType.CLOTH,
        "abilities": {"cloth": {}},
        # Start above the table so it drapes
        "position": [0.2, 0.0, 1.5],
    }

    cfg = dict(scene=scene_cfg, robots=[robot0_cfg], objects=[coffee_table_cfg, cloth_cfg])

    # Create environment (no damage env)
    env = og.Environment(configs=cfg)

    # Robot controls
    robot = env.robots[0]
    controller_choices = {
        "base": "HolonomicBaseJointController",
        "arm_left": "InverseKinematicsController",
        "arm_right": "InverseKinematicsController",
        "gripper_left": "MultiFingerGripperController",
        "gripper_right": "MultiFingerGripperController",
        "camera": "JointController",
        "trunk": "JointController",
    }
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    controller_config["gripper_left"]["inverted"] = True
    controller_config["gripper_right"]["inverted"] = True
    robot.reload_controllers(controller_config=controller_config)
    env.scene.update_initial_file()

    # Viewer camera pose + teleop (W/A/S/D/G)
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.2, -3.2, 2.0]),
        orientation=th.tensor([0.57, 0.10, 0.12, 0.80]),
    )
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

    # Reset env + robot
    env.reset()
    robot.reset()

    # Object refs
    coffee_table = env.scene.object_registry("name", "coffee_table")
    cloth = env.scene.object_registry("name", "cloth")

    # Place cloth on top of the table using OnTop, with manual fallback
    placed = False
    try:
        placed = cloth.states[OnTop].set_value(coffee_table, True)
    except Exception:
        placed = False
    # if not placed:
    #     try:
    #         table_center = coffee_table.aabb_center
    #         table_extent = coffee_table.aabb_extent
    #         top_z = (table_center[2] + 0.5 * table_extent[2]).item()
    #         place_pos = th.tensor([table_center[0].item(), table_center[1].item(), top_z + 0.05])
    #         cloth.set_position_orientation(position=place_pos, orientation=th.tensor([0.0, 0.0, 0.0, 1.0]))
    #     except Exception:
    #         pass

    # Increase cloth damping / stiffness slightly to improve stability
    try:
        cloth.root_link.damping = 0.1  # default ~0.02
    except Exception:
        pass
    try:
        # Mildly increase shear/bend stiffness
        cloth.root_link.shear_stiffness = 90.0
        cloth.root_link.bend_stiffness = 70.0
    except Exception:
        pass

    # Let cloth settle a bit
    for _ in range(60):
        og.sim.step()

    # Impact proxy state (cloth centroid)
    prev_centroid = None
    prev_speed = 0.0

    # Teleop
    action_generator = KeyboardRobotController(robot=robot)
    action_generator.print_keyboard_teleop_info()

    # Loop
    max_steps = 300
    fps = 15

    images = []
    contact_force_proxy = []
    cloth_impact_proxy = []

    tiny = 1e-8
    impact_min_prev_speed = 0.01
    impact_min_decel = 0.01
    # Velocity clamp for cloth particles (m/s) to prevent blow-ups
    vmax = 5.0

    for _ in range(max_steps):
        action = action_generator.get_teleop_action()
        env.step(action=action)

        # Clamp cloth particle velocities to reduce flying away
        try:
            v = cloth.root_link.particle_velocities  # (N,3)
            speeds = th.norm(v, dim=1, keepdim=True)
            mask = speeds > vmax
            if th.any(mask):
                v[mask.squeeze(1)] = v[mask.squeeze(1)] * (vmax / speeds[mask])
                cloth.root_link.particle_velocities = v
        except Exception:
            pass

        # Rigid contact force proxy from table + robot
        table_force = _aggregate_rigid_contact_impulses(coffee_table)
        robot_force = _aggregate_rigid_contact_impulses(robot)
        total_contact_force = table_force + robot_force
        contact_force_proxy.append(total_contact_force)

        # Cloth impact proxy from centroid energy change
        try:
            positions = cloth.root_link.compute_particle_positions()
            centroid = positions.mean(dim=0)
            if prev_centroid is not None:
                disp = centroid - prev_centroid
                # Use disp as velocity proxy per physics step (consistent with mech evaluator)
                vel_t = disp
                speed_t_sq = th.dot(vel_t, vel_t).item()
                # Previous speed was from last step
                speed_prev_sq = prev_speed ** 2
                # Cloth mass; fallback if not available
                try:
                    mass = float(cloth.root_link.mass)
                except Exception:
                    mass = 1.0
                delta_ke_abs = abs(0.5 * mass * (speed_t_sq - speed_prev_sq))
                step_distance = max(th.norm(disp).item(), tiny)
                decel = max(0.0, (prev_speed - (speed_t_sq ** 0.5)))
                if (prev_speed >= impact_min_prev_speed) and (decel >= impact_min_decel):
                    F_impact = delta_ke_abs / step_distance
                else:
                    F_impact = 0.0
                cloth_impact_proxy.append(F_impact)
                prev_speed = (speed_t_sq ** 0.5)
            else:
                prev_speed = 0.0
                cloth_impact_proxy.append(0.0)
            prev_centroid = centroid.clone()
        except Exception:
            cloth_impact_proxy.append(0.0)

        # Frame capture + overlay
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb = rgb.cpu().numpy()[:, :, :3]
        rgb = cv2.resize(rgb, (768, 768))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        y = 32
        cv2.putText(frame, f"Rigid contact force proxy: {total_contact_force:.3f}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y += 32
        last_impact = cloth_impact_proxy[-1] if cloth_impact_proxy else 0.0
        cv2.putText(frame, f"Cloth impact proxy: {last_impact:.3f}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        images.append(np.ascontiguousarray(frame, dtype=np.uint8))

    # Save video
    os.makedirs('videos_and_images', exist_ok=True)
    avi_path = 'videos_and_images/tiago_cloth_mech_teleop.avi'
    mp4_path = 'videos_and_images/tiago_cloth_mech_teleop.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(avi_path, fourcc, fps, (images[0].shape[1], images[0].shape[0]))
    for img in images:
        out.write(img)
    out.release()

    subprocess.run(['ffmpeg', '-y', '-i', avi_path, '-c:v', 'mpeg4', mp4_path], check=True)
    os.remove(avi_path)

    og.clear()
    og.shutdown()


if __name__ == "__main__":
    main() 