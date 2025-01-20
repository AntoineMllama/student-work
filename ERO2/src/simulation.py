import random
from heapq import heappop, heappush

import pandas as pd
import simpy

from consts import SECONDS_PER_HOUR, SECONDS_PER_MINUTE, SECONDS_PER_SECOND
from history import History
from tools import TqdmBar


class Simulation:
    def __init__(
        self, name, population, study, priority, priority_group, tb, servers_amount
    ):
        self.name = name
        self.population = population
        self.study = study
        self.priority = priority
        self.priority_group = priority_group
        self.env = simpy.Environment()
        self.exec_servers = simpy.Resource(self.env, capacity=servers_amount)
        self.result_server = simpy.Resource(self.env, capacity=1)
        self.history = History()
        self.manual_queue = []
        self.soumission_id = 0
        self.blocking_time = tb or 10
        self.t_cycle = self.blocking_time + self.blocking_time / 2
        self.dynamic_priority_index = 0

    def increment_cpt(self):
        while True:
            yield self.env.timeout(self.t_cycle)
            self.dynamic_priority_index += 2

    def add_to_manual_queue(self, submission, current_task):
        arrival_time = self.env.now
        t_mod = self.env.now % self.t_cycle

        self.soumission_id += 1
        priority_values = []

        mapping = {
            "difficulty_level": current_task.difficulty_level,
            "time_remaining": current_task.end_timestamp,
        }

        for field in self.priority:
            if field in mapping:
                if field == "time_remaining":
                    priority_values.append(
                        0
                        if current_task.end_timestamp - self.env.now
                        < SECONDS_PER_MINUTE * 30
                        else 1
                    )
                else:
                    priority_values.append(mapping[field])
            elif field == "population_classifier":
                priority_values.append(
                    self.dynamic_priority_index
                    if submission.task_group == self.priority_group
                    and t_mod < self.blocking_time
                    else self.dynamic_priority_index + 1
                )

        priority_values.append(self.soumission_id)

        priority_tuple = tuple(priority_values)
        heappush(self.manual_queue, (priority_tuple, submission, arrival_time))

    def try_transfer_to_simpy(self):
        while True:
            if len(self.exec_servers.queue) == 0 and len(self.manual_queue) > 0:
                _, submission, arrival_time = heappop(self.manual_queue)
                self.env.process(self.submission_process(submission, arrival_time))
            yield self.env.timeout(0.01)

    def monitor_resources(self, resource, prefix=""):
        while True:
            self.history.add(f"{prefix}utilization_time", self.env.now)
            self.history.add(
                f"{prefix}utilization", len(resource.users) / resource.capacity
            )
            self.history.add(f"{prefix}queue_length_time", self.env.now)
            self.history.add(
                f"{prefix}queue_length", len(resource.queue) + len(self.manual_queue)
            )
            yield self.env.timeout(1)

    def submission_process(self, submission, arrival_time):
        with self.exec_servers.request() as req:  # test suite execution
            yield req
            wait_time_exec = self.env.now - arrival_time
            self.history.add("test_waiting_time", wait_time_exec)
            exec_time = submission.tests_exec_time
            yield self.env.timeout(exec_time)
        with self.result_server.request() as req:  # send result to frontend
            start_wait = self.env.now
            yield req
            wait_time_send = self.env.now - start_wait
            self.history.add("result_waiting_time", wait_time_send)
            send_time = 1
            yield self.env.timeout(send_time)
        departure_time = self.env.now
        self.history.add("arrival_time", arrival_time)
        self.history.add("departure_time", departure_time)
        self.history.add("tag_name", submission.tag_name)
        self.history.add("tag_ok", submission.tag_ok)
        self.history.add("complet", submission.completion)
        self.history.add("solve_time", submission.solve_time)
        self.history.add("spent_time", departure_time - arrival_time)
        self.history.add("exec_step_time", wait_time_exec + exec_time)
        self.history.add("send_step_time", wait_time_send + send_time)
        self.history.add("user_Groupe", submission.task_group.name)

    def work_on_task(self, user, task):
        time = user.seconds_per_task_level[task.difficulty_level]
        if task.name not in user.exo_done:
            # First user needs to use all time to make task
            work_time = time
        else:
            # From the second time, we consider that the user can spend
            # 30 min maximum before making another push
            work_time = min(random.uniform(0, 0.4) * time, SECONDS_PER_MINUTE * 30)
        user.current_work_time += work_time
        return work_time

    def user_process(self, user):
        current_task = None
        need_work_again = False
        solve_time = 0
        while True:
            if user.current_work_time >= user.max_work_duration:
                if random.random() < user.break_probability:
                    break_time = random.uniform(
                        user.break_probability_bounds[0],
                        user.break_probability_bounds[1],
                    )
                    yield self.env.timeout(break_time)
                    user.current_work_time = 0

            if current_task is None:
                available_tasks = self.study.get_available_tasks(user, self.env.now)
                if not available_tasks:
                    if user.in_mental_breakdown:
                        self.env.timeout(SECONDS_PER_HOUR * 24)
                        user.update_mental_state()
                    yield self.env.timeout(1)
                    continue
                current_task = random.choice(available_tasks)
                work_time = self.work_on_task(user, current_task)
                solve_time = work_time
                yield self.env.timeout(work_time)

            if current_task and need_work_again:
                work_time = self.work_on_task(user, current_task)
                solve_time += work_time
                yield self.env.timeout(work_time)
            submission = user.compute_submission(current_task, solve_time)
            is_completed = current_task.check_submission(user, submission, self.env.now)
            self.add_to_manual_queue(submission, current_task)

            if is_completed:
                current_task = None
                need_work_again = False
            else:
                need_work_again = True
            yield self.env.timeout(
                random.uniform(SECONDS_PER_SECOND * 10, SECONDS_PER_MINUTE * 2)
            )

    def run(self):
        self.history.clean()
        self.env.process(self.monitor_resources(self.exec_servers, "exec_"))
        self.env.process(self.monitor_resources(self.result_server, "result_"))
        tqdm_bar = TqdmBar(
            self.study.end_timestamp, self.env, f"Simulation {self.name} en cours"
        )
        self.env.process(tqdm_bar.update_progress())
        self.env.process(self.increment_cpt())
        self.env.process(self.try_transfer_to_simpy())

        for group_users in self.population.users_per_group:
            for user in group_users:
                self.env.process(self.user_process(user))

        self.env.run(until=self.study.end_timestamp)
        tqdm_bar.close()

        df_monitor = pd.DataFrame(
            columns=[
                "exec_utilization_time",
                "exec_utilization",
                "exec_queue_length_time",
                "exec_queue_length",
                "result_utilization_time",
                "result_utilization",
                "result_queue_length_time",
                "result_queue_length",
            ]
        )

        df_submission = pd.DataFrame(
            columns=[
                "tag_name",
                "user_Groupe",
                "tag_ok",
                "complet",
                "solve_time",
                "test_waiting_time",
                "result_waiting_time",
                "spent_time",
                "exec_step_time",
                "send_step_time",
            ]
        )

        df_monitor["exec_utilization_time"] = self.history["exec_utilization_time"]
        df_monitor["exec_utilization"] = self.history["exec_utilization"]
        df_monitor["exec_queue_length_time"] = self.history["exec_queue_length_time"]
        df_monitor["exec_queue_length"] = self.history["exec_queue_length"]

        df_monitor["result_utilization_time"] = self.history["result_utilization_time"]
        df_monitor["result_utilization"] = self.history["result_utilization"]
        df_monitor["result_queue_length_time"] = (self.history)[
            "result_queue_length_time"
        ]
        df_monitor["result_queue_length"] = self.history["result_queue_length"]

        df_submission["tag_name"] = self.history["tag_name"]
        df_submission["tag_ok"] = self.history["tag_ok"]
        df_submission["complet"] = self.history["complet"]
        df_submission["solve_time"] = self.history["solve_time"]
        df_submission["test_waiting_time"] = (self.history)["test_waiting_time"][
            : len(df_submission["solve_time"])
        ]
        df_submission["result_waiting_time"] = self.history["result_waiting_time"][
            : len(df_submission["spent_time"])
        ]
        df_submission["spent_time"] = self.history["spent_time"]
        df_submission["exec_step_time"] = self.history["exec_step_time"]
        df_submission["send_step_time"] = self.history["send_step_time"]
        df_submission["user_Groupe"] = self.history["user_Groupe"]

        return self.history, df_monitor, df_submission
