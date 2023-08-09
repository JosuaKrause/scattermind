from collections.abc import Callable

from scattermind.system.base import ExecutorId
from scattermind.system.executor.executor import Executor, ExecutorManager
from scattermind.system.logger.context import add_context
from scattermind.system.logger.log import EventStream


class SingleExecutorManager(ExecutorManager):
    def __init__(self, own_id: ExecutorId, *, batch_size: int) -> None:
        super().__init__(own_id, batch_size)
        self._is_active = False

    @staticmethod
    def is_local_only() -> bool:
        return True

    def get_all_executors(self) -> list[Executor]:
        return [self.as_executor()]

    def is_active(self, executor_id: ExecutorId) -> bool:
        return self._is_active

    def release_executor(self, executor_id: ExecutorId) -> None:
        pass

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
