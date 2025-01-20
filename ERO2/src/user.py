import random

from entities import TagSubmission


class User:
    def __init__(
        self,
        id,
        group,
        success_per_task_level_probs,
        seconds_per_task_level,
        tag_name_misspell_prob,
        max_work_duration,
        break_probability_bounds,
        break_probability,
        student_level,
        mental_breakdown_prob,
        nb_failure_before_breakdown_per_exo,
    ):
        self.id = id
        self.group = group
        self.success_per_task_level_probs = success_per_task_level_probs
        self.seconds_per_task_level = seconds_per_task_level
        self.tag_name_misspell_prob = tag_name_misspell_prob
        self.max_work_duration = max_work_duration
        self.break_probability_bounds = break_probability_bounds
        self.break_probability = break_probability
        self.student_level = student_level
        self.mental_breakdown_prob = mental_breakdown_prob
        self.nb_failure_before_breakdown_per_exo = nb_failure_before_breakdown_per_exo
        # Liste des exos en cours par l'user, avec son taux de completion
        self.exo_done = {}
        self.exo_failure_nb = {}
        self.current_work_time = 0
        self.in_mental_breakdown = False

    def compute_submission(self, task, solve_time):
        tag_ok = not random.uniform(0.0, 1.0) <= self.tag_name_misspell_prob
        success_rate = self.success_per_task_level_probs[task.difficulty_level]
        # boost de success jusqu'à 50 % selon le taux de réussite de l'exo
        success_rate += (
            0.0 if task.name not in self.exo_done else (self.exo_done[task.name] * 0.5)
        )

        tag_submission = TagSubmission(
            task_name=task.name,
            task_group=task.user_group,
            tag_name=task.tag_prefix,
            tag_ok=tag_ok,
            completion=tag_ok and random.uniform(0.0, 1.0) <= success_rate,
            solve_time=solve_time,
            tests_exec_time=task.tests_exec_time,
        )

        if not tag_submission.completion:
            if task.name not in self.exo_failure_nb:
                self.exo_failure_nb[task.name] = 1
            else:
                self.exo_failure_nb[task.name] += 1
            if (
                self.exo_failure_nb[task.name]
                >= self.nb_failure_before_breakdown_per_exo
                and random.uniform(0.0, 1.0) <= self.mental_breakdown_prob
            ):
                self.in_mental_breakdown = True
        return tag_submission

    def check_dependencies(self, task) -> bool:
        """
        :param task:
        :return: False if the task depends on non-complete dependencies, True otherwise
        """
        return all(self.exo_done.get(dep, False) == 1 for dep in task.dependencies)

    def update_mental_state(self):
        self.in_mental_breakdown = False
        self.exo_failure_nb = {}
