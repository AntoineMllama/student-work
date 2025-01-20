import random

import dtale

from display import display_all_simulations
from metrics import print_metrics
from plot import plot_combined
from population import Population
from simulation import Simulation
from study import Study
from task import Task


class SimulationRunner:
    def __init__(self, simulation_configs):
        self.simulation_configs = simulation_configs
        self.res_df_monitor = []
        self.res_df_submission = []

    def run_simulation(self, simul_name, simul_config):
        print(f"\nRunning Simulation: {simul_name}")
        print(f"RANDOM_STATE: {simul_config.RANDOM_STATE}")
        print(f"SERVERS_AMOUNT: {simul_config.SERVERS_AMOUNT}")
        print(f"Description: {simul_config.description}")
        if simul_config.student_levels:
            print("Student Levels: Enabled")
        else:
            print("Student Levels: Disabled")

        random.seed(simul_config.RANDOM_STATE)

        # Generate population
        population = Population(student_levels_config=simul_config.student_levels)
        population.generate(simul_config.avg_samples_per_group)

        # Create and populate study
        study = Study()
        for task_data in simul_config.tasks:
            task = Task(
                name=task_data.name,
                user_group=task_data.user_group,
                difficulty_level=task_data.difficulty_level,
                tag_prefix=task_data.tag_prefix,
                start_timestamp=task_data.start_timestamp,
                end_timestamp=task_data.end_timestamp,
                rate_limit_hour=task_data.rate_limit_hour,
                rate_limit_day=task_data.rate_limit_day,
                dependencies=task_data.dependencies,
                tests_exec_time=task_data.tests_exec_time,
            )
            study.add_task(task)

        # Run simulation
        simulation = Simulation(
            simul_name,
            population,
            study,
            simul_config.priority,
            simul_config.priority_group,
            simul_config.tb,
            servers_amount=simul_config.SERVERS_AMOUNT,
        )
        history, df_monitor, df_submission = simulation.run()
        self.res_df_monitor.append(df_monitor)
        self.res_df_submission.append(df_submission)

        # Output metrics and plots
        print_metrics(history.history)
        plot_combined(history.history, simul_name, simul_config.description)

        return history

    def run_all_simulations(self):
        simulation_names = []
        metrics_histories = {}

        for simul_name, simul_config in self.simulation_configs.items():
            history = self.run_simulation(simul_name, simul_config)
            simulation_names.append(simul_name)
            metrics_histories[simul_name] = history

        dtale_links = self.load_dtale()
        display_all_simulations(simulation_names, dtale_links, metrics_histories)

    def load_dtale(self):
        all_link = {}
        i = 0

        for simul_name, _ in self.simulation_configs.items():
            df_monitor = self.res_df_monitor[i]
            df_submission = self.res_df_submission[i]

            dtale_web = dtale.show(df_submission)
            dtale.show(df_monitor)
            data_web = f"{dtale_web._url}/dtale/main/"
            chart_web = f"{dtale_web._url}/dtale/charts/"
            data_links = [f"{data_web}{i * 2 + 1}", f"{data_web}{i * 2 + 2}"]
            chart_links = [f"{chart_web}{i * 2 + 1}", f"{chart_web}{i * 2 + 2}"]
            all_link[simul_name] = [data_links, chart_links]
            i += 1

        return all_link
