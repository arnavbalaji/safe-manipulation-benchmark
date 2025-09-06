# Tiago Robot Faucet Teleop Demo

This script demonstrates teleoperating a Tiago robot in an environment with a faucet that can be turned on and off.

## Overview

The demo creates an environment with:
- **Tiago Robot**: A humanoid robot with dual arms and grippers
- **Faucet**: A water faucet that can be toggled on/off (rotated 180° to face the robot)
- **Sink**: A sink basin positioned below the faucet

## Key Features

- **Faucet Control**: Press `F` to toggle the faucet on/off
- **Robot Teleop**: Full control of Tiago's base, arms, and grippers
- **Camera Control**: Move the camera view around the scene
- **State Saving**: Press `TAB` to save the current simulation state
- **Video Recording**: Automatically records and saves videos of the interaction

## Controls

### Robot Control
- **Base Movement**: 
  - `1, 2`: Switch between base joints (x, y, rotation)
  - `[, ]`: Move selected joint backward/forward
- **Arm Control**:
  - Arrow keys: Move arm end-effector
  - `P, ;`: Move arm up/down
  - `N, B`: Rotate arm
  - `O, U`: Rotate arm
  - `V, C`: Rotate arm
- **Gripper Control**:
  - `T`: Toggle gripper open/close

### Special Controls
- `F`: Toggle faucet on/off
- `R`: Reset the robot
- `TAB`: Save simulation state
- `ESC`: Quit the demo

### Camera Control
- `W/A/S/D`: Move camera forward/left/backward/right
- `G`: Move camera down

## Running the Demo

1. Make sure you have the BEHAVIOR-1K environment set up
2. Navigate to the safe-manipulation-benchmark directory
3. Run the script:
   ```bash
   python teleop_tiago_faucet.py
   ```

## What Happens When You Turn On the Faucet

When you press `F` to turn on the faucet:
- The faucet state changes from OFF (0) to ON (1)
- Water particles will start flowing from the faucet spout
- The water will fall into the sink basin
- The faucet state is tracked throughout the simulation

## Output Files

The script generates several video files in the `videos_and_images/` directory:
- `faucet_teleop.mp4`: Main teleop interaction video
- `faucet_state_plot.mp4`: Graph showing faucet state over time
- `faucet_combined_view.mp4`: Side-by-side view of both videos

## Technical Details

- **Faucet Model**: Uses the `zcrgvq` beer_tap model from the BEHAVIOR-1K dataset
- **Sink Model**: Uses the `zexzrc` furniture_sink model (scaled down to 70% for a more compact setup)
- **Physics**: Uses OmniGibson's physics engine with damage simulation
- **Water Simulation**: Leverages OmniGibson's particle system for water effects

## Troubleshooting

### Common Issues

- **Object Loading Errors**: If you get "FileNotFoundError" for USD files, the script will automatically try alternative models
- **Faucet Not Responding**: Make sure the object has the `toggleable` property
- **No Water Flow**: Check that the faucet has the `fluidsource` meta link
- **Performance Issues**: The script enables GPU dynamics (required for water particles) and enables flatcache

### Testing Object Loading

Before running the full demo, you can test if the objects can be loaded:

```bash
python test_faucet_loading.py
```

This will verify that the faucet, sink, and bowl models are available in your BEHAVIOR-1K installation.

### Fallback Options

The script includes multiple fallback configurations:
1. **Primary**: `beer_tap-zcrgvq` + `furniture_sink-zexzrc`
2. **Alternative Sink**: `beer_tap-zcrgvq` + `furniture_sink-czyfhq`
3. **Alternative Faucet**: `beer_tap-vgaluf` + `furniture_sink-zexzrc`
4. **Second Alternative Faucet**: `beer_tap-bebcmz` + `furniture_sink-zexzrc`
5. **Minimal**: `beer_tap-zcrgvq` only
6. **Empty**: Just the Tiago robot (if all else fails)

## References

- Based on `teleop_tiago_bowl.py` and `test_thermal_damage.py`
- Uses the BEHAVIOR-1K knowledge base for object properties
- Implements OmniGibson's object states and particle systems 