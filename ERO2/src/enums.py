from enum import Enum


class UserGroup(int, Enum):
    PREPA1 = 0
    PREPA2 = 1
    ING1 = 2
    ING2 = 3
    ING3 = 4


class TaskDifficultyLevel(int, Enum):
    EASY = 0
    MEDIUM = 1
    HARD = 2


class UserLevel(int, Enum):
    BEGINNER = 0
    INTERMEDIATE = 1
    STRONG = 2
    GEEK = 3


class PriorityType(str, Enum):
    DIFFICULTY_LEVEL = "difficulty_level"
    TIME_REMAINING = "time_remaining"
    POPULATION_CLASSIFIER = "population_classifier"
