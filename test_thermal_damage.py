import torch as th
import cv2
import numpy as np
import os

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of temperature change and thermal damage
    Loads a stove (toggled on) and five apples
    The apples are arranged in a line on the stove, each slightly further from the heat source
    Shows how temperature changes and damage accumulates based on distance from heat source
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Define specific objects we want to load in with the scene directly
    obj_configs = []

    # Light
    obj_configs.append(
        dict(
            type="LightObject",
            light_type="Sphere",
            name="light",
            radius=0.01,
            intensity=1e8,
            position=[-2.0, -2.0, 1.0],
        )
    )

    # Stove
    obj_configs.append(
        dict(
            type="DatasetObject",
            name="stove",
            category="stove",
            model="yhjzwg",
            bounding_box=[1.185, 0.978, 1.387],
            position=[0, 0, 0.69],
        )
    )

    # 5 Apples
    for i in range(5):
        obj_configs.append(
            dict(
                type="DatasetObject",
                name=f"apple{i}",
                category="apple",
                model="agveuv",
                bounding_box=[0.065, 0.065, 0.077],
                position=[0, i * 0.1, 5.0],
                damage_params=PARAMS["apple"],
            )
        )

    # Create the scene config to load -- empty scene with desired objects
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": obj_configs,
    }

    # Create the environment
    env = DamageableEnvironment(configs=cfg)

    # Get reference to relevant objects
    stove = env.scene.object_registry("name", "stove")
    apples = list(env.scene.object_registry("category", "apple"))

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.46938863, -3.97887141, 1.64106008]),
        orientation=th.tensor([0.63311689, 0.00127259, 0.00155577, 0.77405359]),
    )

    # Let objects settle
    for _ in range(25):
        env.step(th.empty(0))

    # Turn on the stove
    stove.states[object_states.ToggledOn].set_value(True)

    # Set initial temperature of the apples to room temperature (20°C)
    for apple in apples:
        apple.states[object_states.Temperature].set_value(20.0)

    # Position the apples in a line, each slightly further from the heat source
    heat_source_pos = stove.states[object_states.HeatSourceOrSink].link.get_position_orientation()[0]
    for i, apple in enumerate(apples):
        apple.set_position_orientation(
            position=heat_source_pos + th.tensor([i * 0.1, i * 0.1, 0.1])  # Move each apple 0.1 units further in x direction
        )

    # Lists to store data for video
    images = []
    temperatures = []
    healths = []

    # Main simulation loop
    print("\nMonitoring apple temperatures and health:")
    print("Step |  0cm  | 10cm  | 20cm  | 30cm  | 40cm")  # Distances from heat source
    print("-" * 45)
    
    for step in range(50):
        env.step(th.empty(0))
        
        # Get RGB image from camera
        rgb_img = og.sim.viewer_camera.get_obs()[0]["rgb"]
        rgb_img = rgb_img.cpu().numpy()[:, :, :3]
        rgb_img = cv2.resize(rgb_img, (512, 512))
        images.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # Get temperatures and health values
        temps = [apple.states[object_states.Temperature].get_value() for apple in apples]
        health_vals = [apple.health for apple in apples]

        temperatures.append(temps)
        healths.append(health_vals)

        if step % 10 == 0:  # Print every 10 steps
            print(f"{step:4d} |" + "|".join(f"{t:6.1f}" for t in temps))

    # Create video
    os.makedirs('videos_and_images', exist_ok=True)
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('videos_and_images/thermal_damage.avi', fourcc, 5.0, (width, height))  # Reduced to 10 fps for slower playback

    # Add data to video frames
    for i, image in enumerate(images):
        frame = image.copy()
        
        # Create a semi-transparent overlay for text background on the right side
        overlay = frame.copy()
        cv2.rectangle(overlay, (width-200, 0), (width, height), (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        y_pos = 30
        font_size = 0.5  # Smaller font size
        font_color = (0, 0, 0)  # Black text
        x_pos = width - 190  # Right side position

        # Add temperature values
        cv2.putText(frame, "Temperatures:", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
        y_pos += 20
        for j, temp in enumerate(temperatures[i]):
            cv2.putText(frame, f"{j*10}cm: {temp:5.1f}", (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            y_pos += 15

        # Add health values
        y_pos += 10
        cv2.putText(frame, "Health Values:", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
        y_pos += 20
        for j, health in enumerate(healths[i]):
            cv2.putText(frame, f"{j*10}cm: {health:5.2f}", (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            y_pos += 15

        out.write(frame)

    out.release()

    # Convert AVI to MP4
    import subprocess
    mp4_path = 'videos_and_images/thermal_damage.mp4'
    subprocess.run([
        'ffmpeg', '-y', '-i', 'videos_and_images/thermal_damage.avi',
        '-c:v', 'mpeg4', mp4_path
    ], check=True)

    # Clean up AVI file
    os.remove('videos_and_images/thermal_damage.avi')

    # Always close env at the end
    og.shutdown()


if __name__ == "__main__":
    main() 