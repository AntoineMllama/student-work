import os
import time

from config_loader import ConfigLoader
from simulation_runner import SimulationRunner


def main():

    # Load configuration
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, "config", "config.json")
    config_loader = ConfigLoader(config_path)
    config = config_loader.get_config()

    # Run all simulations
    runner = SimulationRunner(config)
    runner.run_all_simulations()

    try:  # while True pour que le localhost reste on
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Programme terminé.")


if __name__ == "__main__":
    main()
