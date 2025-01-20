import json
import os

from pydantic import ValidationError

from models_task import SimulationConfig


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data = self.load_and_validate()

    def load_and_validate(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            raw_data = json.load(f)

        # Validate each simulation configuration
        validated_data = {}
        for simul_name, simul_data in raw_data.items():
            try:
                validated_data[simul_name] = SimulationConfig(**simul_data)
            except ValidationError as e:
                raise ValueError(f"Error in simulation '{simul_name}': {e}")

        return validated_data

    def get_config(self):
        return self.config_data
