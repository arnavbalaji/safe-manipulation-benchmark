from safety_benchmark.damage_generators.mechanical_damage_generator import MechanicalDamageGenerator

PARAMS = {
    "bowl": {
        "damage_generators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.0,
            "scale": 6.0,  # Increased scale for more aggressive damage
            "material_properties": {
                "fragility": 10.0,  # Very fragile (glass)
                "elasticity": 0.05,  # Very low bounce (glass doesn't bounce much)
                "density": 2500.0,  # Glass density 
                "contact_threshold": 0.02,  # Very sensitive to impacts
                "energy_threshold": 0.001,  # Very low energy threshold for glass
            }
        }
    },
    "baseball": {
        "damage_generators": ["mechanical"],
        "health_thresholds": [80.0, 50.0, 10.0],
        "mechanical": {
            "damage_threshold": 0.5,
            "scale": 0.01,
            "material_properties": {
                "fragility": 0.5,  # Baseballs are somewhat durable
                "elasticity": 0.8,  # High bounce
                "density": 800.0,  # Leather/rubber density
                "contact_threshold": 0.2,  # Less sensitive to small impacts
                "energy_threshold": 0.02,  # Higher energy threshold
            }
        }
    },
    "coffee_table": {
        "damage_generators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.5,
            "scale": 0.1,
            "material_properties": {
                "fragility": 0.2,  # Tables are very durable
                "elasticity": 0.1,  # Low bounce (wood doesn't bounce)
                "density": 700.0,  # Wood density
                "contact_threshold": 0.5,  # Very insensitive to small impacts
                "energy_threshold": 0.1,  # High energy threshold for wood
            }
        }
    },
    "tiago_robot": {
        "damage_generators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.0,
            "scale": 0.1,
            "material_properties": {
                "fragility": 0.3,  # Robots are moderately durable
                "elasticity": 0.2,  # Low bounce (metal/plastic)
                "density": 1500.0,  # Metal/plastic density
                "contact_threshold": 0.3,  # Moderately sensitive
                "energy_threshold": 0.05,  # Moderate energy threshold
            }
        }
    },
    "default": {
        "damage_generators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.0,
            "scale": 1.0,
            "material_properties": {
                "fragility": 1.0,  # Default fragility
                "elasticity": 0.3,  # Default elasticity
                "density": 1000.0,  # Default density
                "contact_threshold": 0.1,  # Default contact threshold
                "energy_threshold": 0.01,  # Default energy threshold
            }
        }
    }
}

DAMAGE_GENERATORS = {
    "mechanical": MechanicalDamageGenerator
}