from enums import UserGroup


class Study:
    def __init__(self, start_timestamp=0, end_timestamp=0):
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.tasks_per_group = [[] for _ in range(len(UserGroup._member_map_))]

    def add_task(self, task):
        self.tasks_per_group[task.user_group.value].append(task)
        self.start_timestamp = min(self.start_timestamp, task.start_timestamp)
        self.end_timestamp = max(self.end_timestamp, task.end_timestamp)

    def get_available_tasks(self, user, now):
        return [
            task
            for task in self.tasks_per_group[user.group]
            if task.user_group == user.group
            and not task.user_is_done(user)
            and not task.user_is_rate_limited(user, now)
            and task.start_timestamp <= now < task.end_timestamp
            and user.check_dependencies(task)
            and (
                not user.in_mental_breakdown
                or user.exo_failure_nb[task.name]
                >= self.nb_failure_before_breakdown_per_exo
            )
        ]
