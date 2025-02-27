"""
Metaflux Library

Copyright (c) 2025 Lukas G. Olson
Licensed under the MIT License (see LICENSE for details)

================

A lightweight and extensible library for creating, executing, and managing pipelines with dynamic state management,
automatic serialization, retry logic, and resumable execution.

"""

from .pipeline import Pipeline
from .task import Task
from .state import State


from .task_type import TaskType

import logging
logger = logging.getLogger(__name__)