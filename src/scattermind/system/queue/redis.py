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
"""A redis implementation of a queue pool."""
from collections.abc import Callable
from typing import Literal

from redipy import ExecFunction, Redis, RedisConfig
from redipy.api import RSM_MISSING
from redipy.symbolic.expr import Strs
from redipy.symbolic.rlist import RedisList
from redipy.symbolic.rvar import RedisVar
from redipy.symbolic.rzset import RedisSortedSet
from redipy.symbolic.seq import FnContext

from scattermind.system.base import (
    ExecutorId,
    L_REMOTE,
    Locality,
    QueueId,
    TaskId,
)
from scattermind.system.queue.queue import QueuePool


KeyName = Literal[
    "asserts",  # str
    "tasks",  # zset str
    "claims",  # list str
    "expect",  # hash (byte_size, weight)
    "loads",  # set
]
"""Base keys for different storage categories."""


class RedisQueuePool(QueuePool):
    """Redis baked queue pool."""
    def __init__(self, *, cfg: RedisConfig, check_assertions: bool) -> None:
        super().__init__()
        self._redis = Redis("redis", cfg=cfg, redis_module="queues")
        self._check_assertions = check_assertions
        self._claim_tasks = self._claim_tasks_script()

    @staticmethod
    def locality() -> Locality:
        return L_REMOTE

    @staticmethod
    def key(name: KeyName, remain: str) -> str:
        """
        Computes the full key.

        Args:
            name (KeyName): The base key.
            remain (str): The remainder of the key.

        Returns:
            str: The full key.
        """
        return f"{name}:{remain}"

    @classmethod
    def key_assert(cls, task_id: TaskId) -> str:
        """
        Computes the full key for queue assertions. A queue assertion is the
        back link from a task to a queue. This can be used to check whether
        a task is currently in a different queue at the same time (which must
        never happen).

        Args:
            task_id (TaskId): The task id.

        Returns:
            str: The full queue assertion key. The type of the key is a direct
                value.
        """
        return cls.key("asserts", task_id.to_parseable())

    @classmethod
    def key_tasks(cls, qid: QueueId) -> str:
        """
        Computes the full key for task queues.

        Args:
            qid (QueueId): The queue id.

        Returns:
            str: The full task queue key. The type of the key is a zset.
        """
        return cls.key("tasks", qid.to_parseable())

    @classmethod
    def key_claims(cls, qid: QueueId, executor_id: ExecutorId | None) -> str:
        """
        Computes the full key for claims.

        Args:
            qid (QueueId): The queue id.
            executor_id (ExecutorId | None): The claiming executor. If None,
                this segment is ignored.

        Returns:
            str: The full claims key. The type of the key is a list.
        """
        if executor_id is None:
            return cls.key(
                "claims", f"{qid.to_parseable()}:")
        return cls.key(
            "claims", f"{qid.to_parseable()}:{executor_id.to_parseable()}")

    @classmethod
    def key_expect(
            cls, qid: QueueId, field: Literal["byte_size", "weight"]) -> str:
        """
        Computes the full key for expected values.

        Args:
            qid (QueueId): The queue id.
            field (Literal[&quot;byte_size&quot;, &quot;weight&quot;]): Whether
                the value denotes payload size or weight.

        Returns:
            str: The full expected values key. The type of the key is a hash.
        """
        return cls.key(
            "expect", f"{qid.to_parseable()}:{field}")

    @classmethod
    def key_loads(cls, qid: QueueId | None) -> str:
        """
        Computes the full key for loaded queues.

        Args:
            qid (QueueId | None): The queue id or None to obtain the prefix
                pattern only.

        Returns:
            str: The full key. The type of the key is a set.
        """
        return cls.key(
            "loads", "*" if qid is None else f"{qid.to_parseable()}")

    def push_task_id(self, qid: QueueId, task_id: TaskId) -> None:
        # FIXME something better than two connections
        if self._check_assertions:
            assert_key = self.key_assert(task_id)
            if not self._redis.set_value(
                    assert_key, qid.to_parseable(), mode=RSM_MISSING):
                aqid = self._redis.get_value(assert_key)
                raise AssertionError(
                    f"cannot add {task_id} to {qid} because "
                    f"it is already in queue {aqid}")
        with self._redis.pipeline() as pipe:
            weight = self.get_task_weight(task_id)
            pipe.zadd(self.key_tasks(qid), {
                task_id.to_parseable(): weight,
            })

    def get_unclaimed_tasks(self, qid: QueueId) -> list[TaskId]:
        res = self._redis.zrange(self.key_tasks(qid), 0, -1)
        if res is None:
            return []
        return [TaskId.parse(elem) for elem in res]

    def _claim_tasks_script(self) -> ExecFunction:
        ctx = FnContext()
        tasks = RedisSortedSet(ctx.add_key("task_key"))
        claims = RedisList(ctx.add_key("claims_key"))
        assert_key_base = ctx.add_key("assert_key_base")
        qid = ctx.add_arg("qid")
        batch_size = ctx.add_arg("batch_size")
        check_assertions = ctx.add_arg("check_assertions")
        res = ctx.add_local([])
        is_error = ctx.add_local(False)
        aqid = ctx.add_local(None)
        str_help_0 = ctx.add_local("not ")
        str_help_1 = ctx.add_local("")

        # FIXME check elem[0] to elem once in 0.4.0 and check error rendering
        loop, ix, elem = ctx.for_(tasks.pop_max(batch_size))
        n_then, _ = loop.if_(is_error.not_())
        n_then.add(claims.rpush(elem[0]))
        n_then.add(res.set_at(ix, elem[0]))
        a_then, _ = n_then.if_(check_assertions)
        asserts = RedisVar(Strs(assert_key_base, ":", elem[0]))
        a_then.add(aqid.assign(asserts.get_value()))
        a_then.add(asserts.delete())
        e_then, _ = a_then.if_(aqid.ne_(qid))
        e_then.add(is_error.assign(True))
        h_then, _ = e_then.if_(aqid.ne_(None))
        h_then.add(str_help_0.assign(""))
        h_then.add(str_help_1.assign(aqid))
        e_then.add(res.assign(Strs(
            "cannot claim ",
            elem[0],
            " from ",
            qid,
            " because it was ",
            str_help_0,
            "registered in the queue ",
            str_help_1)))
        ctx.set_return_value(res)

        return self._redis.register_script(ctx)

    def claim_tasks(
            self,
            qid: QueueId,
            batch_size: int,
            executor_id: ExecutorId) -> list[TaskId]:
        res = self._claim_tasks(
            keys={
                "task_key": self.key_tasks(qid),
                "claims_key": self.key_claims(qid, executor_id),
                "assert_key_base": "asserts",
            },
            args={
                "qid": qid.to_parseable(),
                "batch_size": batch_size,
                "check_assertions": self._check_assertions,
            })
        if res is None:
            return []
        if isinstance(res, list):
            return [TaskId.parse(elem) for elem in res]
        raise AssertionError(res)

    def claimant_count(self, qid: QueueId) -> int:
        res = self._redis.keys(match=self.key_claims(qid, None), block=False)
        return len(res)

    def unclaim_tasks(
            self, qid: QueueId, executor_id: ExecutorId) -> list[TaskId]:
        claims_key = self.key_claims(qid, executor_id)
        return [
            TaskId.parse(elem)
            for elem in
            self._redis.lrange(claims_key, 0, -1)
        ]

    def get_queue_listeners(self, qid: QueueId) -> int:
        return self._redis.scard(self.key_loads(qid))

    def clean_listeners(self, is_active: Callable[[ExecutorId], bool]) -> int:
        total = 0
        # NOTE: we do not want pipelining/scripting here
        for cur_loads in self._redis.iter_keys(match=self.key_loads(None)):
            to_remove: set[str] = set()
            for executor_id_str in self._redis.smembers(cur_loads):
                remove = False
                try:
                    executor_id = ExecutorId.parse(executor_id_str)
                    remove = not is_active(executor_id)
                except ValueError:
                    remove = True
                if remove:
                    to_remove.add(executor_id_str)
            if to_remove:
                print(f"removed {len(to_remove)} listeners: {to_remove}")
                self._redis.srem(cur_loads, *to_remove)
                total += len(to_remove)
        return total

    def add_queue_listener(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        self._redis.sadd(self.key_loads(qid), executor_id.to_parseable())

    def remove_queue_listener(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        self._redis.srem(self.key_loads(qid), executor_id.to_parseable())

    def expect_task_weight(
            self,
            weight: float,
            byte_size: int,
            qid: QueueId,
            executor_id: ExecutorId) -> None:
        eid = executor_id.to_parseable()
        with self._redis.pipeline() as pipe:
            pipe.hincrby(self.key_expect(qid, "weight"), eid, weight)
            pipe.hincrby(self.key_expect(qid, "byte_size"), eid, byte_size)
            pipe.execute()

    def clear_expected_task_weight(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        eid = executor_id.to_parseable()
        with self._redis.pipeline() as pipe:
            pipe.hdel(self.key_expect(qid, "weight"), eid)
            pipe.hdel(self.key_expect(qid, "byte_size"), eid)
            pipe.execute()

    def get_expected_new_task_weight(self, qid: QueueId) -> float:
        weight = 0.0
        for cweight in self._redis.hvals(self.key_expect(qid, "weight")):
            weight += float(cweight)
        return weight

    def get_expected_byte_size(self, qid: QueueId) -> int:
        byte_size = 0
        for cbyte_size in self._redis.hvals(self.key_expect(qid, "byte_size")):
            byte_size += int(cbyte_size)
        return byte_size

    def get_queue_length(self, qid: QueueId) -> int:
        return self._redis.zcard(self.key_tasks(qid))

    def get_incoming_byte_size(self, qid: QueueId) -> int:
        res = 0
        for task in self.get_unclaimed_compute_tasks(qid):
            res += task.get_byte_size_in()
        return res

    def maybe_get_queue(self, task_id: TaskId) -> QueueId | None:
        res = self._redis.get_value(self.key_assert(task_id))
        if res is None:
            return None
        return QueueId.parse(res)
