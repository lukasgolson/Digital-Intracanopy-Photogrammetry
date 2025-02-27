import inspect

import logging

logger = logging.getLogger(__name__)


class Task:
    """Represents a single task or a group of tasks."""
    def __init__(self, func=None, name=None, **kwargs):
        """
        Initialize a Task instance.
        :param func: Callable task function. If None, the task is treated as a group.
        :param name: Task name.
        :param kwargs: Additional arguments to pass to the task during execution.
        """
        self.func = func
        self.name = name or (func.__name__ if func else "Group")
        self.kwargs = kwargs  # Store additional arguments for the task
        self.is_group = func is None
        self.sub_tasks = []  # Only used if this is a group

    def add_task(self, task):
        """Add a sub-task to the group."""
        if not self.is_group:
            raise ValueError("Cannot add sub-tasks to a non-group task.")
        self.sub_tasks.append(task)

    def run(self, state=None):
        """
        Execute the task or its sub-tasks if it's a group.
        Dynamically passes `state` and `kwargs` based on the callable's signature.
        :param state: Shared state object.
        :return: A tuple (success, result).
        """
        if self.is_group:
            logger.info(f"Running group: {self.name}")
            for task in self.sub_tasks:
                success, _ = task.run(state)
                if not success:
                    logger.error(f"Group {self.name} failed due to sub-task {task.name}.")
                    return False
            return True
        else:
            logger.info(f"Running task: {self.name}")
            try:
                func_signature = inspect.signature(self.func)
                parameters = func_signature.parameters

                args_to_pass = {}
                if 'state' in parameters:
                    args_to_pass['state'] = state
                args_to_pass.update(self.kwargs)

                return self.func(**args_to_pass)
            except Exception as e:
                logger.error(f"Task {self.name} failed with exception: {e}")
                return False
