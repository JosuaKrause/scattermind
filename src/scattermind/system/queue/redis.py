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
                only.

        Returns:
            str: The full key. The type of the key is a set.
        """
        return cls.key(
            "loads", "" if qid is None else f"{qid.to_parseable()}")

    def push_task_id(self, qid: QueueId, task_id: TaskId) -> None:
        # FIXME something better than two connections
        if self._check_assertions:
            assert_key = self.key_assert(task_id)
            if not self._redis.set(
                    assert_key, qid.to_parseable(), mode=RSM_MISSING):
                aqid = self._redis.get(assert_key)
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
        a_then.add(aqid.assign(asserts.get()))
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
        # FIXME: add functionality in redipy
        redis = self._redis.maybe_get_redis_runtime()
        assert redis is not None
        return redis.keys_count(self.key_claims(qid, None))

    def unclaim_tasks(
            self, qid: QueueId, executor_id: ExecutorId) -> list[TaskId]:
        claims_key = self.key_claims(qid, executor_id)
        return [
            TaskId.parse(elem)
            for elem in
            self._redis.lrange(claims_key, 0, -1)
        ]

    def get_queue_listeners(self, qid: QueueId) -> int:
        # FIXME: add sets to redipy
        redis = self._redis.maybe_get_redis_runtime()
        assert redis is not None
        with redis.get_connection() as conn:
            return conn.scard(self.key_loads(qid))

    def clean_listeners(self, is_active: Callable[[ExecutorId], bool]) -> int:
        redis = self._redis.maybe_get_redis_runtime()
        assert redis is not None
        total = 0
        # NOTE: we do not want pipelining/scripting here
        with redis.get_connection() as conn:
            for cur_loads in redis.keys_str(self.key_loads(None)):
                to_remove: set[bytes] = set()
                for executor_id_bytes in conn.smembers(cur_loads):
                    remove = False
                    try:
                        executor_id = ExecutorId.parse(
                            executor_id_bytes.decode("utf-8"))
                        remove = not is_active(executor_id)
                    except ValueError:
                        remove = True
                    if remove:
                        to_remove.add(executor_id_bytes)
                if to_remove:
                    conn.srem(cur_loads, *to_remove)
                    total += len(to_remove)
        return total

    def add_queue_listener(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        # FIXME: add sets to redipy
        redis = self._redis.maybe_get_redis_runtime()
        assert redis is not None
        with redis.get_connection() as conn:
            conn.sadd(self.key_loads(qid), executor_id.to_parseable())

    def remove_queue_listener(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        # FIXME: add sets to redipy
        redis = self._redis.maybe_get_redis_runtime()
        assert redis is not None
        with redis.get_connection() as conn:
            conn.srem(self.key_loads(qid), executor_id.to_parseable())

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
        res = self._redis.get(self.key_assert(task_id))
        if res is None:
            return None
        return QueueId.parse(res)
