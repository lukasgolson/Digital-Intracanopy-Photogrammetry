"""
This module provides a helper function to gracefully handle Ctrl+C (KeyboardInterrupt) signals with many child processes

1. **First Ctrl+C:**
Attempts a *graceful* shutdown of all child processes.
Each child process is asked to terminate, and the program waits briefly.
This allows children to finish their current tasks and exit cleanly.

2. **Second Ctrl+C (or timeout):**
If any child process doesn't respond to the graceful shutdown request
(either by not terminating within a short time or if the user presses Ctrl+C again),
*all remaining child processes are immediately and forcefully terminated*.
This prevents the program from getting stuck if a child process is unresponsive.
"""

import os
import signal
import sys
import psutil


def __find_child_processes():
    """Find child processes of the current process."""
    parent = psutil.Process(os.getpid())
    return parent.children(recursive=True)

def __force_kill_handler(signal_received, frame):
    """Signal handler to forcefully kill child processes."""
    print("Forcefully killing all processes.")
    __terminate_all_children_processes(graceful=False)
    sys.exit(1)


def __terminate_all_children_processes(graceful=True):
    """Terminate or kill all child processes."""
    child_processes = __find_child_processes()
    for child in child_processes:
        try:
            if graceful:
                child.terminate()  # Graceful termination
                child.wait(timeout=5)  # Wait for termination
                print(f"Terminated child process {child.pid}")
            else:
                child.kill()  # Forceful termination
                print(f"Killed child process {child.pid}")
        except psutil.NoSuchProcess:
            print(f"Process {child.pid} already terminated.")
        except psutil.TimeoutExpired:
            print(f"Process {child.pid} did not terminate gracefully. Killing it now.")
            child.kill()  # Force kill if not terminated



def __signal_handler(signal_received, frame):
    """Signal handler for Ctrl+C."""
    print("Ctrl+C detected. Terminating child processes...")
    __terminate_all_children_processes(graceful=True)
    print("Press Ctrl+C again to forcefully kill all processes.")
    signal.signal(signal.SIGINT, __force_kill_handler)  # Register the force kill handler


def register_signal_handlers():
    """Register signal handlers."""
    signal.signal(signal.SIGINT, __signal_handler)