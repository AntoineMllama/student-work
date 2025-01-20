from typing import Dict, List, Optional, Union

from pydantic import BaseModel, root_validator, validator

from enums import PriorityType, TaskDifficultyLevel, UserGroup


class TaskConfig(BaseModel):
    name: str
    user_group: UserGroup
    difficulty_level: TaskDifficultyLevel
    start_timestamp: int
    end_timestamp: int
    tag_prefix: str
    rate_limit_hour: int = 0
    rate_limit_day: int = 0
    dependencies: Optional[List[str]] = None
    tests_exec_time: int

    @validator("user_group", pre=True)
    def validate_user_group(cls, value):
        if isinstance(value, str):
            try:
                return UserGroup[value]
            except KeyError:
                raise ValueError(f"Invalid user group: {value}")
        return value

    @validator("difficulty_level", pre=True)
    def validate_difficulty_level(cls, value):
        if isinstance(value, str):
            try:
                return TaskDifficultyLevel[value]
            except KeyError:
                raise ValueError(f"Invalid difficulty level: {value}")
        return value

    @validator(
        "start_timestamp",
        "end_timestamp",
        "rate_limit_hour",
        "rate_limit_day",
        "tests_exec_time",
    )
    def validate_positive_int(cls, value):
        if value < 0:
            raise ValueError("Timestamp and rate limits must be positive integers.")
        return value


class UserLevelConfig(BaseModel):
    success_probs_modifier: float
    speed_modifier: float
    breakdown_probability: Optional[float] = None

    @validator("success_probs_modifier", "speed_modifier")
    def validate_positive_float(cls, value):
        if value <= 0:
            raise ValueError("Modifiers must be positive")
        return value

    @validator("breakdown_probability")
    def validate_proportion(cls, value):
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"Invalid proportion: f{value}, must be between 0 and 1")
        return value


class UserLevelProportions(BaseModel):
    BEGINNER: float
    INTERMEDIATE: float
    STRONG: float
    GEEK: float

    @validator("BEGINNER", "INTERMEDIATE", "STRONG", "GEEK")
    def validate_proportion(cls, value):
        if not 0 <= value <= 1:
            raise ValueError(f"Invalid proportion: f{value}, must be between 0 and 1")
        return value

    @root_validator
    def validate_total(cls, values):
        total = sum(values.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Sum of proportions must be 1.0, got {total}")
        return values


class UserLevelsSettings(BaseModel):
    modifiers: Dict[str, UserLevelConfig]
    proportions: Dict[str, UserLevelProportions]


class SimulationConfig(BaseModel):
    RANDOM_STATE: int
    SERVERS_AMOUNT: int
    description: Union[str, None]
    avg_samples_per_group: int
    priority: Optional[List[PriorityType]] = []
    priority_group: Optional[UserGroup] = None
    tb: Optional[int] = None
    tasks: List[TaskConfig]
    student_levels: Optional[UserLevelsSettings] = None

    @validator("priority", each_item=True)
    def validate_priority(cls, value):
        if value not in PriorityType:
            raise ValueError(f"Invalid priority type: {value}")
        return value

    @validator("priority_group", pre=True)
    def validate_user_group(cls, value):
        if isinstance(value, str):
            try:
                return UserGroup[value]
            except KeyError:
                raise ValueError(f"Invalid user group: {value}")
        return value

    @validator("tb", pre=True)
    def validate_positive_int(cls, value):
        if value < 0:
            raise ValueError("tb must be positive integers.")
        return value
