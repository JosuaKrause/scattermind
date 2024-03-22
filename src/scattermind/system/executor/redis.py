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
"""A redis based executor that keeps running even if no tasks are available.
"""
import random
import threading
import time
from collections.abc import Callable, Iterator
from typing import Literal

import redis as redis_lib
from redipy import Redis, RedisConfig

from scattermind.system.base import ExecutorId, L_EITHER, Locality
from scattermind.system.executor.executor import Executor, ExecutorManager
from scattermind.system.logger.context import add_context
from scattermind.system.logger.log import EventStream


REDIS_STATUS_PREFIX = "status"


ExecutorState = Literal["running", "request_exit"]
"""The executor state."""
ES_RUN: ExecutorState = "running"
"""The executor is running."""
ES_EXIT: ExecutorState = "request_exit"
"""The executor has been requested to terminate."""


class RedisExecutorManager(ExecutorManager):
    """Manager for a redis based executor that keeps running even if no tasks
    are available."""
    def __init__(
            self,
            own_id: ExecutorId,
            *,
            batch_size: int,
            sleep_on_idle: float,
            reclaim_sleep: float,
            heartbeat_time: float,
            cfg: RedisConfig) -> None:
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
            heartbeat_time (float): The base heartbeat in seconds. The
                expiration is three times the base heartbeat.
            cfg (RedisConfig): The redis configuration.
        """
        super().__init__(own_id, batch_size)
        self._sleep_on_idle = sleep_on_idle
        self._reclaim_sleep = reclaim_sleep
        self._heartbeat_time = heartbeat_time
        self._redis = Redis("redis", cfg=cfg, redis_module="executors")
        self._heartbeat: threading.Thread | None = None
        self._reclaim: threading.Thread | None = None
        self._logger: EventStream | None = None

    def _expire_time(self) -> float:
        expire_time = self._heartbeat_time * 3.0
        if expire_time < 0.0:
            raise ValueError(
                f"invalid expiration time: {self._heartbeat_time}")
        return expire_time

    def _executor_key(self, executor_id: ExecutorId) -> str:
        return f"{REDIS_STATUS_PREFIX}:{executor_id.to_parseable()}"

    def _get_executors(self) -> Iterator[ExecutorId]:
        for key in self._redis.keys(
                match=f"{REDIS_STATUS_PREFIX}:*", block=False):
            eid = key.removeprefix(f"{REDIS_STATUS_PREFIX}:")
            if eid == key:
                raise ValueError(f"invalid key: {key}")
            yield ExecutorId.parse(eid)

    def _set_state(
            self,
            executor_id: ExecutorId,
            state: ExecutorState | None,
            *,
            if_exists: bool) -> None:
        key = self._executor_key(executor_id)
        if state is None:
            self._redis.delete(key)
            return
        self._redis.set_value(
            key,
            state,
            mode="if_exists" if if_exists else "always",
            expire_in=self._expire_time())

    def _get_state(self, executor_id: ExecutorId) -> ExecutorState | None:
        key = self._executor_key(executor_id)
        res = self._redis.get_value(key)
        if res == ES_RUN:
            return ES_RUN
        if res == ES_EXIT:
            return ES_EXIT
        if res is not None:
            self.get_logger().log_warning(
                "warn.executor.state",
                f"invalid state: {res} removing executor {executor_id}")
            self._redis.delete(key)
        return None

    def _refresh_state(self, executor_id: ExecutorId) -> bool:
        key = self._executor_key(executor_id)
        with self._redis.pipeline() as pipe:
            pipe.expire(key, expire_in=self._expire_time())
            pipe.ttl(key)
            _, ttl = pipe.execute()
        return ttl is not None

    def _start_heartbeat(self) -> None:
        logger = self.get_logger()
        own_id = self.get_own_id()

        def run() -> None:
            with add_context({"executor": own_id}):
                running = True
                logger.log_event(
                    "tally.executor.start",
                    {
                        "name": "executor",
                        "action": "heartbeat_start",
                    })
                try:
                    conn_count = 0
                    max_retry = 10
                    while heartbeat is self._heartbeat:
                        hbtime = self._heartbeat_time
                        assert hbtime > 0.0

                        try:
                            if not self._refresh_state(own_id):
                                break
                            time.sleep(hbtime)
                            conn_count = 0
                        except (ConnectionError, redis_lib.ConnectionError):
                            conn_count += 1
                            if conn_count > max_retry:
                                logger.log_error(
                                    "error.executor", "connection")
                            time.sleep(hbtime / max_retry)
                    running = False
                finally:
                    logger.log_event(
                        "tally.executor.stop",
                        {
                            "name": "executor",
                            "action": "heartbeat_stop",
                        })
                    self._heartbeat = None
                    if running:
                        logger.log_error("error.executor", "uncaught_executor")

        if self._heartbeat is not None:
            print("could not start heartbeat. heartbeat already running")
            return
        heartbeat = threading.Thread(
            target=run,
            daemon=True)
        self._heartbeat = heartbeat
        heartbeat.start()

    @staticmethod
    def locality() -> Locality:
        return L_EITHER

    def get_all_executors(self) -> list[Executor]:
        own_id = self.get_own_id()
        return [
            self.as_executor()
            if executor_id == own_id
            else Executor(self, executor_id)
            for executor_id in self._get_executors()
        ]

    def is_active(self, executor_id: ExecutorId) -> bool:
        return self._get_state(executor_id) is not None

    def release_executor(self, executor_id: ExecutorId) -> None:
        self._set_state(executor_id, ES_EXIT, if_exists=True)

    def get_logger(self) -> EventStream:
        """
        Retrieves the logger.

        Returns:
            EventStream: The logger.
        """
        assert self._logger is not None
        return self._logger

    def start_reclaimer(
            self,
            logger: EventStream,
            reclaim_all_once: Callable[[], tuple[int, int]]) -> None:
        own_id = self.get_own_id()

        def do_reclaim() -> None:
            conn_error = 0
            general_error = 0
            while reclaim is self._reclaim:
                try:
                    executor_count, listener_count = reclaim_all_once()
                    if executor_count > 0 or listener_count > 0:
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
                        logger.log_error("error.executor", "connection")
                    time.sleep(60)
                except Exception:  # pylint: disable=broad-exception-caught
                    general_error += 1
                    logger.log_error(
                        "error.executor", "uncaught_executor")
                    if general_error > 10:
                        raise
                    time.sleep(10)
                if not self.is_active(own_id):
                    break
                reclaim_sleep = self._reclaim_sleep
                if reclaim_sleep > 0.0:
                    time.sleep(reclaim_sleep)

        def run() -> None:
            with add_context({"executor": self.get_own_id()}):
                running = True
                try:
                    logger.log_event(
                        "tally.executor.start",
                        {
                            "name": "executor",
                            "action": "reclaim_start",
                        })
                    do_reclaim()
                    running = False
                finally:
                    logger.log_event(
                        "tally.executor.stop",
                        {
                            "name": "executor",
                            "action": "reclaim_stop",
                        })
                    self._reclaim = None
                    if running:
                        logger.log_error("error.executor", "uncaught_executor")

        if self._reclaim is not None:
            print("could not start reclaimer. reclaimer already running")
            return
        reclaim = threading.Thread(
            target=run,
            daemon=True)
        self._reclaim = reclaim
        reclaim.start()

    def execute(
            self,
            logger: EventStream,
            work: Callable[[ExecutorManager], bool]) -> int | None:
        self._logger = logger
        own_id = self.get_own_id()
        running = True
        error = False
        with add_context({"executor": own_id}):
            logger.log_event(
                "tally.executor.start",
                {
                    "name": "executor",
                    "action": "start",
                })
            try:
                conn_init_count = 0
                try:
                    self._set_state(own_id, ES_RUN, if_exists=False)
                except (ConnectionError, redis_lib.ConnectionError):
                    conn_init_count += 1
                    if conn_init_count > 10:
                        logger.log_error("error.executor", "connection")
                    time.sleep(60)
                self._start_heartbeat()
                conn_count = 0
                while self._get_state(own_id) == ES_RUN:
                    sleep_on_idle = self._sleep_on_idle * 0.5
                    sleep_on_idle += random.uniform(
                        0.0, max(sleep_on_idle, 0.0))
                    try:
                        if not work(self) and sleep_on_idle > 0.0:
                            time.sleep(sleep_on_idle)
                        conn_count = 0
                    except (ConnectionError, redis_lib.ConnectionError):
                        conn_count += 1
                        if conn_count > 10:
                            logger.log_error("error.executor", "connection")
                        time.sleep(60)
                running = False
            except Exception:  # pylint: disable=broad-except
                logger.log_error("error.executor", "uncaught_executor")
                error = True
                running = False
            finally:
                self._set_state(own_id, None, if_exists=False)
                logger.log_event(
                    "tally.executor.stop",
                    {
                        "name": "executor",
                        "action": "stop",
                    })
                if running:
                    logger.log_error("error.executor", "uncaught_executor")
        return 1 if error else 0

    @staticmethod
    def allow_parallel() -> bool:
        return False

    def active_count(self) -> int:
        return sum(1 for _ in self._get_executors())
