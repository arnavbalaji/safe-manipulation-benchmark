from safety_benchmark.damage_evaluators.mechanical_damage_evaluator import MechanicalDamageEvaluator
from safety_benchmark.damage_evaluators.thermal_damage_evaluator import ThermalDamageEvaluator

PARAMS = {
    "bowl": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.2,
            "scale": 15.0,  # Increased scale for more aggressive damage
            # "material_properties": {
            #     "fragility": 10.0,  # Very fragile (glass)
            #     "elasticity": 0.05,  # Very low bounce (glass doesn't bounce much)
            #     "density": 2500.0,  # Glass density 
            #     "contact_threshold": 0.02,  # Very sensitive to impacts
            #     "energy_threshold": 0.001,  # Very low energy threshold for glass
            #     "velocity_threshold": 0.05,  # Min velocity change to consider as impact (m/s)
            # }
        }
    },
    "baseball": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [80.0, 50.0, 10.0],
        "mechanical": {
            "damage_threshold": 3.0,
            "scale": 0.01,
            # "material_properties": {
            #     "fragility": 0.5,  # Baseballs are somewhat durable
            #     "elasticity": 0.8,  # High bounce
            #     "density": 800.0,  # Leather/rubber density
            #     "contact_threshold": 0.2,  # Less sensitive to small impacts
            #     "energy_threshold": 0.02,  # Higher energy threshold
            #     "velocity_threshold": 0.05,  # Min velocity change to consider as impact (m/s)
            # }
        }
    },
    "coffee_table": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.5,
            "scale": 0.1,
            # "material_properties": {
            #     "fragility": 0.2,  # Tables are very durable
            #     "elasticity": 0.1,  # Low bounce (wood doesn't bounce)
            #     "density": 700.0,  # Wood density
            #     "contact_threshold": 0.5,  # Very insensitive to small impacts
            #     "energy_threshold": 0.1,  # High energy threshold for wood
            #     "velocity_threshold": 0.05,  # Min velocity change to consider as impact (m/s)
            # }
        }
    },
    "apple": {
        "damage_evaluators": ["thermal"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "thermal": {
            "damage_threshold": 100.0,
            "scale": 0.1,
        }
    },
    "tiago_robot": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 3.0,
            "scale": 0.01,
            "enable_deceleration_detection": False,  # Disable for robots to avoid false positives
            "link_thresholds": {
                "wheel": {"damage_threshold": 5.0, "scale": 0.02},
                "arm": {"damage_threshold": 2.0, "scale": 0.01},
                "gripper": {"damage_threshold": 1.0, "scale": 0.02}
            },
            # "material_properties": {
            #     "fragility": 0.3,  # Robots are moderately durable
            #     "elasticity": 0.2,  # Low bounce (metal/plastic)
            #     "density": 1500.0,  # Metal/plastic density
            #     "contact_threshold": 0.3,  # Moderately sensitive
            #     "energy_threshold": 0.05,  # Moderate energy threshold
            #     "velocity_threshold": 0.05,  # Min velocity change to consider as impact (m/s)
            # }
        }
    },
    "default": {
        "damage_evaluators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.0,
            "scale": 1.0,
            # "material_properties": {
            #     "fragility": 1.0,  # Default fragility
            #     "elasticity": 0.3,  # Default elasticity
            #     "density": 1000.0,  # Default density
            #     "contact_threshold": 0.1,  # Default contact threshold
            #     "energy_threshold": 0.01,  # Default energy threshold
            #     "velocity_threshold": 0.05,  # Min velocity change to consider as impact (m/s)
            # }
        }
    }
}

DAMAGE_EVALUATORS = {
    "mechanical": MechanicalDamageEvaluator,
    "thermal": ThermalDamageEvaluator
}