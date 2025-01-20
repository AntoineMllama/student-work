import random

from consts import SECONDS_PER_HOUR, SECONDS_PER_MINUTE
from enums import UserGroup, UserLevel
from user import User


class Population:
    def __init__(self, student_levels_config=None):
        self.groups_amount = len(UserGroup._member_map_)
        self.users_per_group = [[] for _ in range(self.groups_amount)]
        self.student_levels_config = student_levels_config

    def clean(self):
        self.users_per_group = [[] for _ in range(self.groups_amount)]

    def _check_numeric_bounds(self, bounds, min=float("-inf"), max=float("+inf")):
        assert bounds[0] <= bounds[1]
        assert bounds[0] >= min and bounds[1] <= max
        assert bounds[1] >= min and bounds[1] <= max

    def _check_rate_bounds(self, bounds):
        self._check_numeric_bounds(bounds, 0.0, 1.0)

    def _check_time_bounds(self, bounds):
        self._check_numeric_bounds(bounds, 0, float("+inf"))

    def _create_student_levels_list(self, total_students, group):
        if not (
            self.student_levels_config
            and self.student_levels_config.proportions
            and group.name in self.student_levels_config.proportions
        ):
            return [None] * total_students

        proportions = self.student_levels_config.proportions[group.name]
        student_levels = []

        for level, proportion in proportions.dict().items():
            nb_students = round(total_students * proportion)
            student_levels.extend([UserLevel[level]] * nb_students)

        while len(student_levels) < total_students:
            student_levels.append(UserLevel.INTERMEDIATE)
        while len(student_levels) > total_students:
            student_levels.pop()

        random.shuffle(student_levels)
        return student_levels

    def _adjust_by_user_level(
        self,
        group,
        student_level,
        base_success_probs,
        base_seconds,
        base_mental_breakdown_prob,
    ):
        if not (
            self.student_levels_config
            and student_level
            and self.student_levels_config.proportions
            and group.name in self.student_levels_config.proportions
        ):
            return base_success_probs, base_seconds, base_mental_breakdown_prob

        modifiers = self.student_levels_config.modifiers[student_level.name]
        success_probs = [
            min(0.99, prob * modifiers.success_probs_modifier)
            for prob in base_success_probs
        ]
        seconds = [
            max(SECONDS_PER_MINUTE, int(time * (1 / modifiers.speed_modifier)))
            for time in base_seconds
        ]
        if modifiers.breakdown_probability:
            return success_probs, seconds, modifiers.breakdown_probability
        return success_probs, seconds, base_mental_breakdown_prob

    def generate(
        self,
        avg_samples_per_group,
        samples_per_group_deviation=0.1,
        success_rate_bounds_easy_task=(0.7, 1.0),
        success_rate_bounds_medium_task=(0.4, 0.8),
        success_rate_bounds_hard_task=(0.1, 0.5),
        seconds_bounds_easy_task=(SECONDS_PER_MINUTE * 10, SECONDS_PER_MINUTE * 15),
        seconds_bounds_medium_task=(SECONDS_PER_MINUTE * 30, SECONDS_PER_HOUR),
        seconds_bounds_hard_task=(SECONDS_PER_HOUR * 2, SECONDS_PER_HOUR * 4),
        misspell_rate_bounds=(0.0, 0.05),
        max_work_duration=SECONDS_PER_HOUR * 2,
        break_probability_bounds=(SECONDS_PER_MINUTE * 5, SECONDS_PER_MINUTE * 25),
        break_probability=(0.6, 1.0),
    ):
        self._check_rate_bounds(success_rate_bounds_easy_task)
        self._check_rate_bounds(success_rate_bounds_medium_task)
        self._check_rate_bounds(success_rate_bounds_hard_task)
        self._check_rate_bounds(misspell_rate_bounds)
        self._check_time_bounds(seconds_bounds_easy_task)
        self._check_time_bounds(seconds_bounds_medium_task)
        self._check_time_bounds(seconds_bounds_hard_task)
        self._check_time_bounds(break_probability_bounds)
        self._check_rate_bounds(break_probability)

        assert samples_per_group_deviation > 0.0 and samples_per_group_deviation <= 1.0
        self.clean()
        next_id = 0
        for group_ii in range(self.groups_amount):
            samples_amount = random.randint(
                round(avg_samples_per_group * (1 - samples_per_group_deviation)),
                round(avg_samples_per_group * (1 + samples_per_group_deviation)),
            )
            student_levels = self._create_student_levels_list(
                samples_amount, UserGroup(group_ii)
            )

            for user_ii in range(samples_amount):
                (
                    success_per_task_level_probs,
                    seconds_per_task_level,
                    mental_breakdown_prob,
                ) = self._adjust_by_user_level(
                    UserGroup(group_ii),
                    student_levels[user_ii],
                    base_success_probs=[
                        random.uniform(*success_rate_bounds_easy_task),
                        random.uniform(*success_rate_bounds_medium_task),
                        random.uniform(*success_rate_bounds_hard_task),
                    ],
                    base_seconds=[
                        random.randint(*seconds_bounds_easy_task),
                        random.randint(*seconds_bounds_medium_task),
                        random.randint(*seconds_bounds_hard_task),
                    ],
                    base_mental_breakdown_prob=random.uniform(0.01, 0.2),
                )

                user = User(
                    id=next_id,
                    group=UserGroup(group_ii),
                    success_per_task_level_probs=success_per_task_level_probs,
                    seconds_per_task_level=seconds_per_task_level,
                    tag_name_misspell_prob=random.uniform(*misspell_rate_bounds),
                    max_work_duration=max_work_duration,
                    break_probability_bounds=break_probability_bounds,
                    break_probability=random.uniform(*break_probability),
                    student_level=student_levels[user_ii],
                    mental_breakdown_prob=mental_breakdown_prob,
                    nb_failure_before_breakdown_per_exo=15,
                )
                self.users_per_group[group_ii].append(user)
                next_id += 1
        return self.users_per_group

    def add_user(self, user):
        self.users_per_group[user.group.value].append(user)
