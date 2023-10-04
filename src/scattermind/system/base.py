import uuid
from collections.abc import Callable
from typing import TypeVar

from scattermind.system.names import GName, NAME_SEP, NName


SelfT = TypeVar('SelfT', bound='BaseId')
"""A `BaseId` subclass."""

NS_GRAPH = uuid.UUID("15f123374f9f4a9385a5f31b6ed3f630")
"""Namespace for graph ids."""
NS_NODE = uuid.UUID("3446718135514d6c9ea36b5377ea0a19")
"""Namespace for node ids."""
NS_QUEUE = uuid.UUID("55fbaa9565fa4bf3ba806ade03c1e395")
"""Namespace for queue ids."""
UUID_OUTPUT_QUEUE = uuid.UUID("8a82aef7da2c4b49bf7ba25d42e99391")
"""Output queue id."""
TEST_EXECUTOR = uuid.UUID("5883f8c8a5bc4d56ae3202d1dc548118")


DEBUG_OUTPUT_LENGTH: int | None = None


def set_debug_output_length(output_length: int | None) -> None:
    global DEBUG_OUTPUT_LENGTH

    DEBUG_OUTPUT_LENGTH = output_length


def trim_id(text_id: str) -> str:
    if DEBUG_OUTPUT_LENGTH is None:
        return text_id
    return text_id[:DEBUG_OUTPUT_LENGTH]


class BaseId:
    """
    A `UUID` based id. The id consists of a single character prefix to
    distinguish different id types and a uuid.
    """
    def __init__(self, raw_id: uuid.UUID) -> None:
        """
        Creates an id. Do not call this function directly.
        Use `parse` or the respective `create` instead.

        Args:
            raw_id (uuid.UUID): The UUID.
        """
        self._id = raw_id

    @staticmethod
    def prefix() -> str:
        """
        The prefix to distinguish different id types.

        Returns:
            str: A single uppercase character prefix that is unique and
            identifies the given type.
        """
        raise NotImplementedError()

    @classmethod
    def parse(cls: type[SelfT], text: str) -> SelfT:
        """
        Parses a string into an id.
        Use `to_parseable` to obtain a parseable string.

        Args:
            cls (type[SelfT]): The id class.
            text (str): The parseable string.

        Raises:
            ValueError: If the string did not represent an id.

        Returns:
            SelfT: The id.
        """
        if not text.startswith(cls.prefix()):
            raise ValueError(f"invalid prefix for {cls.__name__}: {text}")
        if len(text) != 33 or "-" in text:
            raise ValueError(f"invalid {cls.__name__}: {text}")
        return cls(uuid.UUID(text[1:]))

    def _hex(self) -> str:
        return self._id.hex

    def to_parseable(self) -> str:
        """
        Creates a parseable representation of the id.

        Returns:
            str: A string that can be parsed by `parse`.
        """
        return f"{self.prefix()}{self._hex()}"

    def __eq__(self, other: object) -> bool:
        """
        Tests whether two id objects are the same.

        Args:
            other (object): The other id object.

        Returns:
            bool: Id objects are equal if they have the same hash and prefix.
        """
        if other is self:
            return True
        if not isinstance(other, BaseId):
            return False
        if self.prefix() != other.prefix():
            return False
        return self._id == other._id

    def __ne__(self, other: object) -> bool:
        """
        Tests whether two id objects are not the same.

        Args:
            other (object): The other id object.

        Returns:
            bool: Id objects are unequal if they have different hash or prefix.
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """
        The hash of an id.

        Returns:
            int: The hash only depends on the id hash (not the prefix).
        """
        return hash(self._id)

    def __str__(self) -> str:
        return f"{self.prefix()}[{trim_id(self._hex())}]"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.__str__()}]"


class GraphId(BaseId):
    @staticmethod
    def prefix() -> str:
        return "G"

    @staticmethod
    def create(gname: GName) -> 'GraphId':
        """
        Creates a `GraphId` from a given graph name.

        Args:
            gname (GName): The graph name.

        Returns:
            GraphId: The corrseponding graph id.
        """
        return GraphId(uuid.uuid5(NS_GRAPH, gname.get()))


class NodeId(BaseId):
    """
    A `NodeId` is used to identify a node in the execution graph.
    """
    @staticmethod
    def prefix() -> str:
        return "N"

    @staticmethod
    def create(gname: GName, name: NName) -> 'NodeId':
        """
        Creates a `NodeId` from a given node name.

        Args:
            gname (GName): The graph name.
            name (NName): The node name.

        Returns:
            NodeId: The corrseponding node id.
        """
        return NodeId(
            uuid.uuid5(NS_NODE, f"{gname.get()}{NAME_SEP}{name.get()}"))


class QueueId(BaseId):
    """
    A `QueueId` is used to identify a queue in the execution graph.
    """
    @staticmethod
    def prefix() -> str:
        return "Q"

    @staticmethod
    def create(gname: GName, node_name: NName) -> 'QueueId':
        """
        Creates a `QueueId` for a given graph name and node name.

        Args:
            gname (GName): The graph name.
            node_name (NName): The node name.

        Returns:
            QueueId: The corrseponding queue id.
        """
        return QueueId(
            uuid.uuid5(NS_QUEUE, f"{gname.get()}{NAME_SEP}{node_name.get()}"))

    @staticmethod
    def get_output_queue() -> 'QueueId':
        """
        The `QueueId` for the result queue with the output of an
        execution graph.
        """
        return OUTPUT_QUEUE

    def is_output_id(self) -> bool:
        """
        Whether the queue id represents the result queue id.

        Returns:
            bool: `True` if the id equals `OUTPUT_QUEUE`.
        """
        return self == OUTPUT_QUEUE

    def ensure_queue(self) -> 'QueueId':
        """
        Ensures that the given queue id is associated with a queue and not
        the output id.

        Raises:
            ValueError: If the queue id is the output id.

        Returns:
            QueueId: The queue id.
        """
        if self.is_output_id():
            raise ValueError("queue id is output id")
        return self


OUTPUT_QUEUE = QueueId(UUID_OUTPUT_QUEUE)
"""The `QueueId` for the result queue with the output of an execution graph."""


class TaskId(BaseId):
    """
    A `TaskId` is used to identify a task throughout its execution.
    """
    @staticmethod
    def prefix() -> str:
        return "T"

    @staticmethod
    def create() -> 'TaskId':
        """
        Creates a `TaskId`.

        Returns:
            TaskId: A unique `TaskId`.
        """
        return TaskId(uuid.uuid4())


class ExecutorId(BaseId):
    """
    An `ExecutorId` is used to identify an executor. When starting, an executor
    gets a unique id assigned.
    """
    @staticmethod
    def prefix() -> str:
        return "E"

    @staticmethod
    def create() -> 'ExecutorId':
        """
        Creates a `ExecutorId`.

        Returns:
            ExecutorId: A unique `ExecutorId`.
        """
        return ExecutorId(uuid.uuid4())

    @staticmethod
    def for_test() -> 'ExecutorId':
        return TEST_EXECUTOR_ID


TEST_EXECUTOR_ID = ExecutorId(TEST_EXECUTOR)


def once_test_executors(execs: list[ExecutorId]) -> Callable[[], ExecutorId]:
    teix = 0

    def gen() -> ExecutorId:
        nonlocal teix

        ix = teix
        teix += 1
        if ix < len(execs):
            return execs[ix]
        raise ValueError(
            f"generator can only produce each value once: {ix} {execs}")

    return gen


SelfD = TypeVar('SelfD', bound='DataId')  # pylint: disable=invalid-name
"""A `DataId` subclass."""


class DataId:
    """
    A data id is used to identify a piece of data in temporary storage.
    The exact format of a data id is left to the storage to decide.
    """
    def __init__(self, raw_id: str) -> None:
        """
        Creates a `DataId`.

        Args:
            raw_id (str): The raw id string.
        """
        self._id = raw_id

    def raw_id(self) -> str:
        """
        Returns the raw id string.

        Returns:
            str: The raw id string.
        """
        return self._id

    @staticmethod
    def prefix() -> str:
        """
        The prefix to distinguish different id types.

        Returns:
            str: A single uppercase character prefix that is unique and
            identifies the given type.
        """
        return "D"

    @staticmethod
    def validate_id(raw_id: str) -> bool:
        """
        Determines whether a raw id string represents a valid data id.

        Args:
            raw_id (str): The raw id string.

        Returns:
            bool: `True` if the raw id string is valid.
        """
        raise NotImplementedError()

    @classmethod
    def parse(cls: type[SelfD], text: str) -> SelfD:
        """
        Parses a string into an id.
        Use `to_parseable` to obtain a parseable string.

        Args:
            cls (type[SelfD]): The id class.
            text (str): The parseable string.

        Raises:
            ValueError: If the string did not represent an id.

        Returns:
            SelfD: The id.
        """
        if not text.startswith(cls.prefix()):
            raise ValueError(f"invalid prefix for {cls.__name__}: {text}")
        raw_id = text[1:]
        if not cls.validate_id(raw_id):
            raise ValueError(f"invalid id for {cls.__name__}: {text}")
        return cls(raw_id)

    def to_parseable(self) -> str:
        """
        Creates a parseable representation of the id.

        Returns:
            str: A string that can be parsed by `parse`.
        """
        return f"{self.prefix()}{self._id}"

    def __eq__(self, other: object) -> bool:
        """
        Tests whether two id objects are the same.

        Args:
            other (object): The other id object.

        Returns:
            bool: Id objects are equal if they have the same raw id and prefix.
        """
        if other is self:
            return True
        if not isinstance(other, DataId):
            return False
        return self._id == other._id

    def __ne__(self, other: object) -> bool:
        """
        Tests whether two id objects are not the same.

        Args:
            other (object): The other id object.

        Returns:
            bool: Id objects are unequal if their raw ids or prefixes differ.
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """
        The hash of an id.

        Returns:
            int: The hash only depends on the id hash (not the prefix).
        """
        return hash(self._id)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self.to_parseable()}]"

    def __repr__(self) -> str:
        return self.__str__()


class Module:  # pylint: disable=too-few-public-methods
    """
    A module for environment dependent behavior. Module classes need to be
    subclassed to implement the respective behavior.
    """
    @staticmethod
    def is_local_only() -> bool:
        """
        Whether the module is for a local (same process) environment only.
        All loaded modules must be either all local or all non-local.

        Returns:
            bool: `True` if the module is local only.
        """
        raise NotImplementedError()
