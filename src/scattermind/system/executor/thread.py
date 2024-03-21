# Copyright (C) 2024 Josua Krause
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A thread based executor that keeps running even if no tasks are available.
"""
import random
import sys
import threading
import time
from collections.abc import Callable

import redis as redis_lib

from scattermind.system.base import ExecutorId, L_EITHER, Locality
from scattermind.system.executor.executor import Executor, ExecutorManager
from scattermind.system.logger.context import add_context
from scattermind.system.logger.log import EventStream


LOCK = threading.RLock()
"""The main lock."""
EXECUTORS: dict[ExecutorId, 'ThreadExecutorManager'] = {}
"""All registered executors."""
ACTIVE: dict[ExecutorId, bool] = {}
"""Indicates which executors are active."""
ALIVE: set[ExecutorId] = set()
"""The set of executors that are alive. This number is not reliable for
accounting for unexpectedly terminated executors. Use listener count for this.
"""


class ThreadExecutorManager(ExecutorManager):
    """Manager for a thread based executor that keeps running even if no tasks
    are available. The number of threads is specified in the configuration and
    the corresponding number of managers will be created."""
    def __init__(
            self,
            own_id: ExecutorId,
            *,
            batch_size: int,
            sleep_on_idle: float,
            reclaim_sleep: float) -> None:
        """
        Creates an executor manager for the given executor id.

        Args:
            own_id (ExecutorId): The executor id.
            batch_size (int): The batch size for processing tasks of the
                given executor.
            sleep_on_idle (float): The time to sleep in seconds if no task
                is currently available for processing.
            reclaim_sleep (float): The time to sleep in seconds between
                two reclaim calls.
        """
        super().__init__(own_id, batch_size)
        self._sleep_on_idle = sleep_on_idle
        self._reclaim_sleep = reclaim_sleep
        self._thread: threading.Thread | None = None
        self._reclaim: threading.Thread | None = None
        self._work: Callable[[ExecutorManager], bool] | None = None
        self._logger: EventStream | None = None
        self._done = False
        with LOCK:
            EXECUTORS[own_id] = self
            ACTIVE[own_id] = True

    def is_done(self) -> bool:
        """
        Whether the executor should terminate at its earliest convenience.

        Returns:
            bool: If True, the executor should terminate when possible.
        """
        return self._done

    def set_done(self, is_done: bool) -> None:
        """
        Mark the executor for termination.

        Args:
            is_done (bool): If True, the executor will terminate when possible.
        """
        self._done = is_done

    def _maybe_start(self) -> None:
        # FIXME: detect when all workers terminated and exit the program with
        # the correct status code
        assert self._logger is not None
        logger = self._logger

        def run() -> None:
            with add_context({"executor": self.get_own_id()}):
                running = True
                logger.log_event(
                    "tally.executor.start",
                    {
                        "name": "executor",
                        "action": "start",
                    })
                try:
                    with LOCK:
                        ALIVE.add(self.get_own_id())
                    conn_count = 0
                    while not self.is_done() and thread is self._thread:
                        work = self._work
                        if work is None:
                            raise ValueError("uninitialized executor")
                        sleep_on_idle = self._sleep_on_idle * 0.5
                        sleep_on_idle += random.uniform(
                            0.0, max(sleep_on_idle, 0.0))
                        try:
                            if not work(self) and sleep_on_idle > 0.0:
                                time.sleep(sleep_on_idle)
                        except (ConnectionError, redis_lib.ConnectionError):
                            conn_count += 1
                            if conn_count > 10:
                                logger.log_error(
                                    "error.executor", "connection")
                            time.sleep(60)
                    running = False
                finally:
                    logger.log_event(
                        "tally.executor.stop",
                        {
                            "name": "executor",
                            "action": "stop",
                        })
                    with LOCK:
                        ALIVE.discard(self.get_own_id())
                        ACTIVE[self.get_own_id()] = False
                        self._thread = None
                        if running:
                            logger.log_error(
                                "error.executor", "uncaught_executor")
                            sys.exit(1)

        if self._thread is not None:
            return
        with LOCK:
            if self._thread is not None:
                return
            if self.is_done() or not self.is_active(self.get_own_id()):
                raise ValueError("attempt to start inactive executor")
            thread = threading.Thread(
                target=run,
                daemon=False)
            self._thread = thread
            thread.start()

    @staticmethod
    def locality() -> Locality:
        return L_EITHER  # FIXME: figure out a way to clearly define it here

    def get_all_executors(self) -> list[Executor]:
        with LOCK:
            return [emng.as_executor() for emng in EXECUTORS.values()]

    def is_active(self, executor_id: ExecutorId) -> bool:
        with LOCK:
            return ACTIVE.get(executor_id, False)

    def release_executor(self, executor_id: ExecutorId) -> None:
        with LOCK:
            executor = EXECUTORS[executor_id]
            if executor is not None:
                executor.set_done(True)

    def get_work(self) -> Callable[[ExecutorManager], bool] | None:
        """
        Retrieves the currently installed work callback.

        Returns:
            Callable[[ExecutorManager], bool] | None: The work callback or
                None if the executor has not been started yet.
        """
        return self._work

    def get_logger(self) -> EventStream | None:
        """
        Retrieves the logger.

        Returns:
            EventStream | None: The logger or None if no logger has been set.
        """
        return self._logger

    def start_reclaimer(
            self,
            logger: EventStream,
            reclaim_all_once: Callable[[], tuple[int, int]]) -> None:
        conn_error = 0
        general_error = 0

        def reclaim() -> None:
            nonlocal conn_error
            nonlocal general_error

            try:
                executor_count, listener_count = reclaim_all_once()
                if executor_count or listener_count:
                    logger.log_event(
                        "tally.executor.reclaim",
                        {
                            "name": "executor",
                            "action": "reclaim",
                            "executors": executor_count,
                            "listeners": listener_count,
                        })
                conn_error = 0
                general_error = 0
            except (ConnectionError, redis_lib.ConnectionError):
                conn_error += 1
                if conn_error > 10:
                    logger.log_error(
                        "error.executor", "connection")
                time.sleep(60)
            except Exception:  # pylint: disable=broad-exception-caught
                general_error += 1
                if general_error > 10:
                    raise
                logger.log_error("error.executor", "uncaught_executor")
                time.sleep(10)
            reclaim_sleep = self._reclaim_sleep
            if reclaim_sleep > 0.0:
                time.sleep(reclaim_sleep)

        def run() -> None:
            with add_context({"executor": self.get_own_id()}):
                try:
                    logger.log_event(
                        "tally.executor.start",
                        {
                            "name": "executor",
                            "action": "reclaim_start",
                        })
                    while True:
                        reclaim()
                finally:
                    logger.log_event(
                        "tally.executor.stop",
                        {
                            "name": "executor",
                            "action": "reclaim_stop",
                        })
                    with LOCK:
                        self._reclaim = None
                        logger.log_error("error.executor", "uncaught_executor")

        if self._reclaim is not None:
            return
        with LOCK:
            if self._reclaim is not None:
                return
            thread = threading.Thread(
                target=run,
                daemon=True)
            self._reclaim = thread
            thread.start()

    def execute(  # pylint: disable=useless-return
            self,
            logger: EventStream,
            work: Callable[[ExecutorManager], bool]) -> int | None:
        if self._work is not work or self._logger is not logger:
            self._work = work
            self._logger = logger
            self._maybe_start()
            # NOTE: propagate the work to all other thread executors
            with LOCK:
                executors = list(EXECUTORS.values())
            for exe in executors:
                is_active = exe.is_active(exe.get_own_id())
                same_work = exe.get_work() is work
                same_logger = self.get_logger() is logger
                if (not same_work or not same_logger) and is_active:
                    if exe.execute(logger, work) is not None:
                        raise RuntimeError("this should not happen!")
        return None

    @staticmethod
    def allow_parallel() -> bool:
        return True

    def active_count(self) -> int:
        return len(ALIVE)
