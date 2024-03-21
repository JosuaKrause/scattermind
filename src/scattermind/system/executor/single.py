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
"""A singular executor that terminates when all tasks are processed."""
from collections.abc import Callable

from scattermind.system.base import ExecutorId, L_EITHER, Locality
from scattermind.system.executor.executor import Executor, ExecutorManager
from scattermind.system.logger.context import add_context
from scattermind.system.logger.log import EventStream


class SingleExecutorManager(ExecutorManager):
    """Manager for a singular executor that terminates when all tasks are
    processed."""
    def __init__(self, own_id: ExecutorId, *, batch_size: int) -> None:
        super().__init__(own_id, batch_size)
        self._is_active = False

    @staticmethod
    def locality() -> Locality:
        return L_EITHER  # FIXME: figure out a way to clearly define it here

    def get_all_executors(self) -> list[Executor]:
        return [self.as_executor()]

    def is_active(self, executor_id: ExecutorId) -> bool:
        return self._is_active

    def release_executor(self, executor_id: ExecutorId) -> None:
        pass

    def start_reclaimer(
            self,
            logger: EventStream,
            reclaim_all_once: Callable[[], tuple[int, int]]) -> None:
        pass  # NOTE: we do not reclaim executors

    def execute(
            self,
            logger: EventStream,
            work: Callable[[ExecutorManager], bool]) -> int | None:
        with add_context({"executor": self.get_own_id()}):
            running = True
            error = False
            logger.log_event(
                "tally.executor.start",
                {
                    "name": "executor",
                    "action": "start",
                })
            try:
                self._is_active = True
                done = False
                while not done:
                    done = not work(self)
                running = False
            except Exception:  # pylint: disable=broad-except
                logger.log_error("error.executor", "uncaught_executor")
                running = False
                error = True
            finally:
                self._is_active = False
                logger.log_event(
                    "tally.executor.stop",
                    {
                        "name": "executor",
                        "action": "stop",
                    })
                if running:
                    logger.log_error("error.executor", "uncaught_executor")
            return 1 if running or error else 0

    @staticmethod
    def allow_parallel() -> bool:
        return False

    def active_count(self) -> int:
        return 1 if self._is_active else 0
