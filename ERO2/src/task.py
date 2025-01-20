import random
from collections import deque

from consts import SECONDS_PER_HOUR
from enums import TaskDifficultyLevel, UserGroup


class Task:
    def __init__(
        self,
        name,
        user_group=UserGroup.ING1,
        difficulty_level=TaskDifficultyLevel.MEDIUM,
        start_timestamp=0,
        end_timestamp=0,
        tag_prefix="",
        rate_limit_hour=0,
        rate_limit_day=0,
        dependencies=None,
        tests_exec_time=2,
    ):
        self.name = name
        self.user_group = user_group
        self.difficulty_level = difficulty_level
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.tag_prefix = tag_prefix
        self.rate_limit_hour = rate_limit_hour
        self.rate_limit_day = rate_limit_day
        self.dependencies = [] if dependencies is None else dependencies
        self.tests_exec_time = tests_exec_time
        self.submission_history = {}
        self.users_done = {}

    def user_is_rate_limited(self, user, now):
        if user.id not in self.submission_history:
            self.submission_history[user.id] = deque()
            return False
        else:
            one_hour_ago = now - SECONDS_PER_HOUR
            one_day_ago = now - SECONDS_PER_HOUR * 24
            self.submission_history[user.id] = deque(
                timestamp
                for timestamp in self.submission_history[user.id]
                if timestamp >= one_day_ago
            )
            submissions_last_day = len(self.submission_history[user.id])
            submissions_last_hour = sum(
                1 for ts in self.submission_history[user.id] if ts >= one_hour_ago
            )
            is_rate_limited = False
            if self.rate_limit_hour > 0:
                is_rate_limited = submissions_last_hour >= self.rate_limit_hour
            if self.rate_limit_day > 0:
                is_rate_limited = (
                    is_rate_limited or submissions_last_day >= self.rate_limit_day
                )
            return is_rate_limited

    def user_is_done(self, user):
        return user.id in self.users_done and self.users_done[user.id]

    def check_submission(self, user, submission, now):
        if user.id not in self.users_done:
            self.users_done[user.id] = False
        elif self.users_done[user.id]:
            return True

        if submission.tag_ok:
            if submission.completion:
                self.users_done[user.id] = True
                user.exo_done[submission.task_name] = 1
            else:
                if submission.task_name not in user.exo_done:  # First Tag
                    user.exo_done[submission.task_name] = random.uniform(0.0, 0.99)
                else:
                    # Tag x, on assumera dans un premier temps que l'avancement de l'exo
                    # augmente uniquement (ne peut pas passer de 72% à 42%)
                    user.exo_done[submission.task_name] = random.uniform(
                        user.exo_done[submission.task_name], 0.99
                    )
        self.submission_history[user.id].append(now)
        return self.users_done[user.id]
