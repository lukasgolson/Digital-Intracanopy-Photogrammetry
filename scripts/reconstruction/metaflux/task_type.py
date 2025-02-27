from enum import Enum


class TaskType(Enum):
    """
    Enum to define the different types of tasks within a pipeline.

    Attributes:
        INIT: Represents initialization tasks that are executed before the main tasks.
              These tasks typically set up the environment or perform prerequisite operations.
        MAIN: Represents the core tasks of the pipeline, forming the primary workflow.
              These are the main steps that achieve the pipeline's objectives.
        INTER: Represents intermediary tasks that are executed between main tasks.
               These tasks are often used for transitions, validations, or intermediate processes
               like document saving.
        RETRY: Represents tasks that are executed when a main task fails and is retried.
    """
    INIT = "initialization"  # Tasks to prepare the pipeline before main execution.
    MAIN = "main"            # Core tasks forming the primary workflow of the pipeline.
    INTER = "intermediary"   # Tasks executed between main tasks for additional processing.
    RETRY = "retry"          # Tasks that are executed when a main task fails and is retried. Useful for state recovery.
