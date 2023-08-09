import threading

from scattermind.system.base import ExecutorId, QueueId, TaskId
from scattermind.system.logger.context import ctx_fmt
from scattermind.system.queue.queue import QueuePool


class LocalQueuePool(QueuePool):
    def __init__(self, check_assertions: bool) -> None:
        super().__init__()
        self._check_assertions = check_assertions
        self._assert_tasks: dict[TaskId, QueueId] = {}
        self._task_ids: dict[QueueId, list[TaskId]] = {}
        self._claims: dict[ExecutorId, dict[QueueId, list[TaskId]]] = {}
        self._expect: dict[QueueId, dict[ExecutorId, tuple[float, int]]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def is_local_only() -> bool:
        return True

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
            task_ids.append(task_id)

    def get_unclaimed_tasks(self, qid: QueueId) -> list[TaskId]:
        with self._lock:
            return list(self._task_ids.get(qid, []))

    def claim_tasks(
            self,
            qid: QueueId,
            batch_size: int,
            executor_id: ExecutorId) -> list[TaskId]:
        with self._lock:
            task_ids = self._task_ids.get(qid, [])
            self.sort_tasks(task_ids)
            qclaims = self._claims.get(executor_id)
            if qclaims is None:
                qclaims = {}
                self._claims[executor_id] = qclaims
            claims = qclaims.get(qid)
            if claims is None:
                claims = []
                qclaims[qid] = claims
            res: list[TaskId] = []
            while task_ids and len(res) < batch_size:
                task_id = task_ids.pop(0)
                claims.append(task_id)
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

    def unclaim_tasks(
            self, qid: QueueId, executor_id: ExecutorId) -> list[TaskId]:
        with self._lock:
            qclaims = self._claims.get(executor_id, {})
            return qclaims.pop(qid, [])

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

    def get_queue_length(self, qid: QueueId) -> int:
        with self._lock:
            return len(self._task_ids.get(qid, []))

    def get_expected_byte_size(self, qid: QueueId) -> int:
        with self._lock:
            qmeta = self._expect.get(qid, {})
            byte_size = 0
            for _, cbyte_size in qmeta.values():
                byte_size += cbyte_size
            return byte_size

    def get_incoming_byte_size(self, qid: QueueId) -> int:
        cpool = self.get_client_pool()
        with self._lock:
            res = 0
            for task_id in self._task_ids.get(qid, []):
                task = self.get_compute_task(cpool, qid, task_id)
                res += task.get_byte_size_in()
            return res

    def maybe_get_queue(self, task_id: TaskId) -> QueueId | None:
        with self._lock:
            return self._assert_tasks.get(task_id)
