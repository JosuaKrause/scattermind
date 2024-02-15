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
"""A RAM-only implementation of a queue pool."""
import threading
from collections.abc import Callable

from scattermind.system.base import (
    ExecutorId,
    L_LOCAL,
    Locality,
    QueueId,
    TaskId,
)
from scattermind.system.logger.context import ctx_fmt
from scattermind.system.queue.queue import QueuePool


class LocalQueuePool(QueuePool):
    """A RAM-only implementation of a queue pool."""
    def __init__(self, *, check_assertions: bool) -> None:
        """
        Creates a local queue pool.

        Args:
            check_assertions (bool): Whether to check assertions (what queue
                is a given task in?).
        """
        super().__init__()
        self._check_assertions = check_assertions
        self._assert_tasks: dict[TaskId, QueueId] = {}
        self._task_ids: dict[QueueId, list[tuple[float, TaskId]]] = {}
        self._claims: dict[QueueId, dict[ExecutorId, list[TaskId]]] = {}
        self._loads: dict[QueueId, set[ExecutorId]] = {}
        self._expect: dict[QueueId, dict[ExecutorId, tuple[float, int]]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def locality() -> Locality:
        return L_LOCAL

    def push_task_id(self, qid: QueueId, task_id: TaskId) -> None:
        with self._lock:
            print(f"{ctx_fmt()} add {task_id} to {qid}")
            if self._check_assertions:
                assert_tasks = self._assert_tasks
                aqid = assert_tasks.get(task_id)
                if aqid is not None:
                    raise AssertionError(
                        f"cannot add {task_id} to {qid} because "
                        f"it is already in queue {aqid}")
                assert_tasks[task_id] = qid
            task_ids = self._task_ids.get(qid)
            if task_ids is None:
                task_ids = []
                self._task_ids[qid] = task_ids
            weight = self.get_task_weight(task_id)
            task_ids.append((weight, task_id))

    def get_unclaimed_tasks(self, qid: QueueId) -> list[TaskId]:
        with self._lock:
            return [task_id for _, task_id in self._task_ids.get(qid, [])]

    def claim_tasks(
            self,
            qid: QueueId,
            batch_size: int,
            executor_id: ExecutorId) -> list[TaskId]:
        with self._lock:
            task_ids = self._task_ids.get(qid, [])
            # FIXME: keep a sorted list instead of sorting every time
            task_ids.sort(reverse=True, key=lambda elem: elem[0])
            claims = self._claims.get(qid)
            if claims is None:
                claims = {}
                self._claims[qid] = claims
            qclaims = claims.get(executor_id)
            if qclaims is None:
                qclaims = []
                claims[executor_id] = qclaims
            res: list[TaskId] = []
            while task_ids and len(res) < batch_size:
                _, task_id = task_ids.pop(0)
                qclaims.append(task_id)
                res.append(task_id)
            if self._check_assertions:
                assert_tasks = self._assert_tasks
                for task_id in res:
                    aqid = assert_tasks.pop(task_id, None)
                    if aqid != qid:
                        raise AssertionError(
                            f"cannot claim {task_id} from {qid} because it "
                            f"was {'not ' if aqid is None else ''}registered "
                            "in the queue "
                            f"{'' if aqid is None else aqid}".rstrip())
            self._task_ids[qid] = task_ids
            print(f"{ctx_fmt()} claim {res} in {qid} remaining {task_ids}")
            return res

    def claimant_count(self, qid: QueueId) -> int:
        with self._lock:
            qclaims = self._claims.get(qid, {})
            return len(qclaims)

    def unclaim_tasks(
            self, qid: QueueId, executor_id: ExecutorId) -> list[TaskId]:
        with self._lock:
            qclaims = self._claims.get(qid, {})
            return qclaims.pop(executor_id, [])

    def get_queue_listeners(self, qid: QueueId) -> int:
        with self._lock:
            return len(self._loads.get(qid, set()))

    def clean_listeners(self, is_active: Callable[[ExecutorId], bool]) -> int:
        with self._lock:
            loads = list(self._loads.values())
        total = 0
        for load_val in loads:
            to_remove: list[ExecutorId] = []
            with self._lock:
                for executor_id in load_val:
                    if is_active(executor_id):
                        continue
                    to_remove.append(executor_id)
            with self._lock:
                for executor_id in to_remove:
                    load_val.discard(executor_id)
            total += len(to_remove)
        return total

    def add_queue_listener(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        with self._lock:
            loads = self._loads.get(qid)
            if loads is None:
                loads = set()
                self._loads[qid] = loads
            loads.add(executor_id)

    def remove_queue_listener(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        with self._lock:
            loads = self._loads.get(qid)
            if loads is not None:
                loads.discard(executor_id)

    def expect_task_weight(
            self,
            weight: float,
            byte_size: int,
            qid: QueueId,
            executor_id: ExecutorId) -> None:
        with self._lock:
            qmeta = self._expect.get(qid)
            if qmeta is None:
                qmeta = {}
                self._expect[qid] = qmeta
            cweight, cbyte_size = qmeta.get(executor_id, (0.0, 0))
            cweight += weight
            cbyte_size += byte_size
            qmeta[executor_id] = (cweight, cbyte_size)

    def clear_expected_task_weight(
            self, qid: QueueId, executor_id: ExecutorId) -> None:
        with self._lock:
            qmeta = self._expect.get(qid)
            if qmeta is not None:
                qmeta.pop(executor_id, 0.0)

    def get_expected_new_task_weight(self, qid: QueueId) -> float:
        with self._lock:
            qmeta = self._expect.get(qid, {})
            weight = 0.0
            for cweight, _ in qmeta.values():
                weight += cweight
            return weight

    def get_expected_byte_size(self, qid: QueueId) -> int:
        with self._lock:
            qmeta = self._expect.get(qid, {})
            byte_size = 0
            for _, cbyte_size in qmeta.values():
                byte_size += cbyte_size
            return byte_size

    def get_queue_length(self, qid: QueueId) -> int:
        with self._lock:
            return len(self._task_ids.get(qid, []))

    def get_incoming_byte_size(self, qid: QueueId) -> int:
        cpool = self.get_client_pool()
        with self._lock:
            res = 0
            for _, task_id in self._task_ids.get(qid, []):
                task = self.get_compute_task(cpool, qid, task_id)
                res += task.get_byte_size_in()
            return res

    def maybe_get_queue(self, task_id: TaskId) -> QueueId | None:
        with self._lock:
            return self._assert_tasks.get(task_id)
