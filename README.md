# Improved Mechanical Damage System for OmniGibson

## Overview

This improved mechanical damage system provides realistic physics-based damage calculation for objects in OmniGibson simulations. Unlike the previous implementation that only tracked contact forces, this system calculates damage based on **impact energy**, **velocity changes**, and **material properties**.

## Key Features

### 🎯 Physics-Based Damage Calculation
- **Impact Energy**: Damage scales with kinetic energy before impact (KE = 0.5 × mass × velocity²)
- **Velocity Changes**: Detects sudden deceleration indicating impacts
- **Material Properties**: Different materials have different fragility, elasticity, and damage thresholds

### 🔍 Smart Impact Detection
- **Impact vs Static Contact**: Distinguishes between actual impacts and normal resting contact
- **Velocity Thresholds**: Only considers significant velocity changes as impacts
- **Contact State Tracking**: Monitors when objects transition from no-contact to contact

### 🏗️ Material-Specific Properties
- **Fragility**: How easily objects break (0.1 = very fragile, 10.0 = very tough)
- **Elasticity**: How much energy is absorbed vs transferred (0.0 = no bounce, 1.0 = perfect bounce)
- **Density**: Affects impact energy calculation
- **Contact Thresholds**: Minimum velocity change to consider as impact
- **Energy Thresholds**: Minimum kinetic energy to cause damage

## How It Works

### 1. Impact Detection
The system tracks:
- Previous and current velocities for each object link
- Contact state changes (no contact → contact)
- Velocity change magnitude

An impact is detected when:
```python
is_impact = (
    velocity_change > contact_threshold and
    not prev_contacts and current_contacts and
    prev_velocity_magnitude > 0.1  # Was moving before impact
)
```

### 2. Damage Calculation
Damage is calculated using:
```python
impact_energy = 0.5 * mass * velocity²
base_damage = impact_energy * scale
damage = base_damage * fragility * (1 - elasticity)
```

### 3. Material Properties
Each object can have custom material properties:
```python
material_properties = {
    "fragility": 5.0,        # Wine glasses are very fragile
    "elasticity": 0.1,       # Low bounce (glass doesn't bounce much)
    "density": 2500.0,       # Glass density (kg/m³)
    "contact_threshold": 0.05,  # Very sensitive to impacts
    "energy_threshold": 0.005,  # Low energy threshold for glass
}
```

## Usage

### Basic Setup

1. **Import the damage system**:
```python
from safety_benchmark.damageable_env import DamageableEnvironment
from safety_benchmark.params.test_params import PARAMS
```

2. **Create objects with damage parameters**:
```python
wine_glass_config = {
    "type": "DatasetObject",
    "name": "wine_glass",
    "category": "wineglass",
    "model": "cmdagy",
    "damage_params": PARAMS["bowl"],  # Uses wine glass material properties
    "mass": 0.2,
}
```

3. **Create the environment**:
```python
env = DamageableEnvironment(configs=cfg)
env.reset()
```

### Monitoring Damage

```python
# Get object health
wine_glass = env.scene.object_registry("name", "wine_glass")
health = wine_glass.health
status = wine_glass.damage_status

# Get impact history for debugging
impact_history = wine_glass.get_impact_history()
```

### Custom Material Properties

You can define custom material properties for any object:

```python
custom_params = {
    "damage_generators": ["mechanical"],
    "health_thresholds": [90.0, 60.0, 30.0],
    "mechanical": {
        "damage_threshold": 0.0,
        "scale": 1.0,
        "material_properties": {
            "fragility": 2.0,        # Custom fragility
            "elasticity": 0.5,       # Custom elasticity
            "density": 1500.0,       # Custom density
            "contact_threshold": 0.1, # Custom contact threshold
            "energy_threshold": 0.01, # Custom energy threshold
        }
    }
}
```

## Examples

### Wine Glass Drop Scenario
```python
# Wine glass sitting on table (no damage)
wine_glass.set_position_orientation(position=[0, 0, 0.5])
# Let physics settle - no damage should occur

# Drop wine glass from height (significant damage)
wine_glass.set_position_orientation(position=[0, 0, 1.0])
# Let it fall and impact - damage should occur based on impact energy
```

### Material Comparison
```python
# Drop wine glass and baseball from same height
wine_glass.set_position_orientation(position=[0, 0, 1.0])
baseball.set_position_orientation(position=[0.5, 0, 1.0])

# Wine glass should take more damage (more fragile)
wine_damage = 100.0 - wine_glass.health
baseball_damage = 100.0 - baseball.health
# wine_damage > baseball_damage
```

## Testing

Run the test suite to verify the system works correctly:

```bash
cd safety_benchmark
python test_improved_damage.py
```

This will test:
- ✅ Static contact doesn't cause damage
- ✅ Drop impacts cause realistic damage
- ✅ Damage scales with impact velocity
- ✅ Different materials have different damage responses

## Demo

Run the interactive demo to see the system in action:

```bash
cd safety_benchmark
python demo_improved_damage.py
```

Controls:
- `H` - Print health status
- `I` - Print impact history
- `R` - Reset environment
- `D` - Drop wine glass
- `B` - Drop baseball
- `T` - Drop both objects

## Configuration

### Default Material Properties

| Object Type | Fragility | Elasticity | Density | Contact Threshold | Energy Threshold |
|-------------|-----------|------------|---------|-------------------|------------------|
| Wine Glass  | 5.0       | 0.1        | 2500    | 0.05              | 0.005            |
| Baseball    | 0.5       | 0.8        | 800     | 0.2               | 0.02             |
| Coffee Table| 0.2       | 0.1        | 700     | 0.5               | 0.1              |
| Robot       | 0.3       | 0.2        | 1500    | 0.3               | 0.05             |

### Health Thresholds
- **Minor**: 90.0 - First damage threshold
- **Major**: 60.0 - Significant damage threshold  
- **Critical**: 30.0 - Severe damage threshold

## Troubleshooting

### Common Issues

1. **No damage from drops**: Check that material properties have appropriate fragility and energy thresholds
2. **Damage from static contact**: Verify contact thresholds are set appropriately
3. **Inconsistent damage**: Ensure mass values are realistic for the object size

### Debugging

Use the impact history to debug damage calculations:

```python
impact_history = obj.get_impact_history()
for link_name, impacts in impact_history.items():
    if impacts:
        latest = impacts[-1]
        print(f"Impact energy: {latest['impact_energy']:.3f}J")
        print(f"Velocity change: {latest['velocity_change']:.2f}m/s")
```

## Physics Background

The system is based on real physics principles:

1. **Kinetic Energy**: KE = 0.5 × mass × velocity²
2. **Impact Force**: F = mass × acceleration (velocity change / time)
3. **Material Stress**: Stress = Force / Area (simplified to force for this system)

The damage calculation approximates how real materials respond to impacts, considering:
- Energy absorption (elasticity)
- Material strength (fragility)
- Impact severity (energy and velocity thresholds)

## Future Improvements

Potential enhancements:
- **Stress Distribution**: Consider impact location and object geometry
- **Fatigue Damage**: Accumulated damage from repeated impacts
- **Temperature Effects**: Material properties changing with temperature
- **Strain Rate Effects**: Different damage responses at different impact speeds
- **Fracture Mechanics**: Realistic crack propagation and failure modes 