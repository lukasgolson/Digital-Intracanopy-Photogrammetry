from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Union, Callable, List

from .state import State
from .task import Task
from .task_type import TaskType

import logging

logger = logging.getLogger(__name__)

class Pipeline:
    """Manages and executes a sequence of tasks or groups."""

    def __init__(self, name="Pipeline", max_retries=2):

        """
        Creates a new pipeline instance.
        :param name: The name of the pipeline, for identification purposes.
        :param max_retries:  The maximum number of retries for each task or group.
        """

        self.name = name
        self.init_tasks = []
        self.tasks = []
        self.inter_tasks = []
        self.retry_tasks = []
        self.max_retries = max_retries
        self.current_task_index = 0
        self.pipeline_hash = None

    def add_task(
            self,
            func_or_task_or_list: Union[Task, Pipeline, Callable, List[Union[Task, Pipeline, Callable]]],
            task_type: TaskType = TaskType.MAIN, **kwargs
    ) -> Pipeline:
        """
        Add a task, group, or multiple tasks to the pipeline.
        Automatically wraps functions in a Task object if necessary.

        :param task_type: The type of task to add (main, intermediary, or initialization).
        :param func_or_task_or_list: A Task object, a Pipeline object,
                                     a callable function, or a list of these.
        :param kwargs: Additional arguments to pass to the Task if a function is provided.

        :return: The Pipeline instance.
        """

        # validate the task type
        if not isinstance(task_type, TaskType):
            raise ValueError("task_type must be an instance of TaskType.")

        # Select the task list based on the task type
        task_list = {
            TaskType.INIT: self.init_tasks,
            TaskType.MAIN: self.tasks,
            TaskType.INTER: self.inter_tasks,
            TaskType.RETRY: self.retry_tasks
        }.get(task_type)

        if isinstance(func_or_task_or_list, list):
            for item in func_or_task_or_list:
                self.add_task(item, task_type, **kwargs)  # Recursive call for individual tasks
        elif isinstance(func_or_task_or_list, (Task, Pipeline)):
            task_list.append(func_or_task_or_list)

        elif callable(func_or_task_or_list):
            task_list.append(Task(func=func_or_task_or_list, **kwargs))
        else:
            raise ValueError("Argument must be a Task, a Pipeline, a callable function, or a list of these.")

        self.pipeline_hash = self.compute_pipeline_hash()

        return self

    def compute_pipeline_hash(self):
        """
        Compute a hash for the pipeline based on its tasks.
        Ensures consistency between saved and restored pipelines.
        """
        pipeline_structure = [task.name if isinstance(task, Task) else task.name for task in self.tasks]
        pipeline_str = json.dumps(pipeline_structure, sort_keys=True)
        return hashlib.sha256(pipeline_str.encode()).hexdigest()

    def save(self, filepath, state):
        """
        Save the current pipeline progress and state to a file.
        :param filepath: Filepath to save the pipeline state.
        :param state: The current state object.
        """
        data = {
            "current_task_index": self.current_task_index,
            "pipeline_hash": self.pipeline_hash,
            "state": state.to_dict(),
        }
        with open(filepath, "wb") as file:
            pickle.dump(data, file)

    @staticmethod
    def load(filepath):
        """
        Load a saved pipeline state from a file.
        :param filepath: Filepath to load the pipeline state from.
        :return: A tuple of (current_task_index, pipeline_hash, State).
        """
        with open(filepath, "rb") as file:
            data = pickle.load(file)
        state = State.from_dict(data["state"])
        return data["current_task_index"], data["pipeline_hash"], state

    def validate_pipeline_hash(self, saved_hash):
        """
        Validate the pipeline hash against the saved hash.
        :param saved_hash: The hash from the saved pipeline state.
        :return: True if the hash matches, False otherwise.
        """
        return self.pipeline_hash == saved_hash

    def run(self, state=None, resume_filepath=None):
        """
        Execute all tasks in this pipeline, respecting task types.
        """

        if resume_filepath and Path(resume_filepath).exists():
            saved_index, saved_hash, loaded_state = self.load(resume_filepath)
            if not self.validate_pipeline_hash(saved_hash):
                raise ValueError("Pipeline structure has changed since the last save.")
            self.current_task_index = saved_index
            state = loaded_state
            logger.info(f"Resuming pipeline instance: {self.name} with {len(self.tasks)} tasks." )
        else:
            logger.info(f"Creating new pipeline instance: {self.name} with {len(self.tasks)} tasks.")
            state = state or State()

        if not isinstance(state, State):
            raise ValueError("state must be an instance of State")

        if not self._run_tasks(self.init_tasks, state, "Starting Tasks"):
            logger.info("Starting tasks failed. Aborting pipeline.")
            return False

        # Execute main tasks with intermediary tasks in between
        for i in range(self.current_task_index, len(self.tasks)):
            task = self.tasks[i]
            retries = 0
            while retries < self.max_retries:
                if isinstance(task, Pipeline):
                    logger.info(f"Running child pipeline: {task.name}")
                    success = task.run(state=state)
                else:
                    success = task.run(state)

                if success:
                    logger.info(f"Task {task.name} completed successfully.")
                    self.current_task_index += 1
                    if resume_filepath is not None:
                        self.save(resume_filepath, state)

                    if not self._run_tasks(self.inter_tasks, state, "Intermediary Tasks"):
                        return False

                    break
                else:
                    retries += 1
                    logger.warning(f"Task {task.name} failed ({retries}/{self.max_retries}). Running retry tasks...")

                    self._run_tasks(self.retry_tasks, state, "Retry Tasks")


                    logger.info(f"Retrying task {task.name} ({retries}/{self.max_retries})...")


            if retries >= self.max_retries:
                logger.error(f"Task {task.name} failed after {self.max_retries} retries. Skipping.")
                return False

        return True

    def _run_tasks(self, tasks: List[Task], state: State, task_type: str) -> bool:
        """
        Run a list of tasks sequentially with retries.
        :param tasks: List of tasks to execute.
        :param state: The shared state object.
        :param task_type: A string describing the task type for logging purposes.
        """
        logger.info(f"Running {task_type}...")
        for task in tasks:
            retries = 0
            while retries < self.max_retries:
                success = task.run(state)
                if success:
                    logger.info(f"{task_type} task {task.name} completed successfully.")
                    break
                retries += 1
                logger.info(f"Retrying {task_type} task {task.name} ({retries}/{self.max_retries})...")

            if retries >= self.max_retries:
                logger.info(f"{task_type} task {task.name} failed after {self.max_retries} retries.")
                return False  # Fail if any task fails

        return True

    def map_pipeline(self) -> dict:
        """
        Generate a hierarchical map of the pipeline structure,
        including initialization tasks, main tasks, and intermediary tasks,
        and marking the resumption point.
        :return: A dictionary representing the pipeline structure.
        """

        def map_task(task):
            if isinstance(task, Pipeline):
                # Recursively map sub-pipelines
                return {"type": "Pipeline", "name": task.name, "tasks": task.map_pipeline()}
            elif isinstance(task, Task):
                return {"type": "Task", "name": task.name}
            else:
                return {"type": "Unknown", "details": str(task)}

        return {
            "name": self.name,
            "init_tasks": [map_task(task) for task in self.init_tasks],
            "main_tasks": [
                {"task": map_task(task), "resume_point": i == self.current_task_index}
                for i, task in enumerate(self.tasks)
            ],
            "inter_tasks": [map_task(task) for task in self.inter_tasks],
        }

    def get_pipeline_string(self):
        """
        Print a formatted view of the pipeline structure,
        showing initialization tasks, main tasks, and intermediary tasks,
        and marking the resumption point with a symbol for intermediaries.
        """

        def format_task(task, indent=0):
            spacer = "  " * indent
            if task["type"] == "Pipeline":
                result = f"{spacer}- {task['name']} (Pipeline):\n"
                for sub_task in task["tasks"]:
                    result += format_task(sub_task, indent + 1)
                return result
            elif task["type"] == "Task":
                return f"{spacer}- {task['name']} (Task)\n"
            else:
                return f"{spacer}- {task.get('details', 'Unknown Task')}\n"

        def format_section(title, tasks, indent=0):
            spacer = "  " * indent
            section_str = f"{spacer}{title}:\n"
            for task in tasks:
                if isinstance(task, dict) and "resume_point" in task:
                    # Main task with resume point indicator
                    section_str += f"{spacer}  - {task['task']['name']} (Task)"
                    if task["resume_point"]:
                        section_str += " [Resume Point]"
                    section_str += "\n"
                else:
                    # Other task types (e.g., initialization or intermediary)
                    section_str += format_task(task, indent + 1)
            return section_str

        def format_main_with_intermediaries(main_tasks, intermediaries, indent=0):
            spacer = "  " * indent
            result = f"{spacer}Main Tasks with Intermediaries:\n"
            for i, task in enumerate(main_tasks):
                # Format main task
                result += f"{spacer}  - {task['task']['name']} (Task)"
                if task["resume_point"]:
                    result += " [Resume Point]"
                result += "\n"

                # Add intermediary symbol and tasks if not the last main task
                if i < len(main_tasks) - 1 and intermediaries:
                    result += f"{spacer}    --> Intermediary Tasks:\n"
                    for inter_task in intermediaries:
                        result += format_task(inter_task, indent + 3)
            return result

        pipeline_map = self.map_pipeline()
        output = f"Pipeline: {pipeline_map['name']}\n"
        output += format_section("Initialization Tasks", pipeline_map["init_tasks"], indent=1)
        output += format_main_with_intermediaries(pipeline_map["main_tasks"], pipeline_map["inter_tasks"], indent=1)
        return output
