# Scattermind distributes computation of machine learning models.
# Copyright (C) 2024 Josua Krause
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
        return L_EITHER

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
            work: Callable[[ExecutorManager], bool]) -> None:
        with add_context({"executor": self.get_own_id()}):
            running = True
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
            finally:
                self._is_active = False
                if running:
                    logger.log_error("error.executor", "uncaught_executor")
                logger.log_event(
                    "tally.executor.stop",
                    {
                        "name": "executor",
                        "action": "stop",
                    })

    @staticmethod
    def allow_parallel() -> bool:
        return False
