from safety_benchmark.damage_generators.mechanical_damage_generator import MechanicalDamageGenerator

PARAMS = {
    "bowl": {
        "damage_generators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.0,
            "scale": 3.0
        }
    },
    "baseball": {
        "damage_generators": ["mechanical"],
        "health_thresholds": [80.0, 50.0, 10.0],
        "mechanical": {
            "damage_threshold": 0.5,
            "scale": 0.01
        }
    },
    "coffee_table": {
        "damage_generators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.5,
            "scale": 0.1
        }
    },
    "tiago_robot": {
        "damage_generators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.0,
            "scale": 0.1
        }
    },
    "default": {
        "damage_generators": ["mechanical"],
        "health_thresholds": [90.0, 60.0, 30.0],
        "mechanical": {
            "damage_threshold": 0.0,
            "scale": 1.0
        }
    }
}

DAMAGE_GENERATORS = {
    "mechanical": MechanicalDamageGenerator
}