from dataclasses import dataclass

from enums import UserGroup


@dataclass
class TagSubmission:
    task_name: str
    task_group: UserGroup
    tag_name: str
    tag_ok: bool = True
    completion: bool = False
    solve_time: float = 0
    tests_exec_time: float = 0
