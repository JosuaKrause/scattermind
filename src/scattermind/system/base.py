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
"""This module defines various ids used internally. Also various other base
classes are defined here."""
import os
import uuid
from collections.abc import Callable, Iterable
from typing import Literal, TypeVar

import torch

from scattermind.system.io import get_subfolders
from scattermind.system.names import NAME_SEP, NName, QualifiedGraphName, UName
from scattermind.system.torch_util import (
    create_tensor,
    tensor_list,
    tensor_to_str,
)


SelfT = TypeVar('SelfT', bound='BaseId')
"""A `BaseId` subclass."""

NS_USER = uuid.UUID("f41a0ce6a69e485d9c26f03ed4671203")
"""Namespace for user ids."""
NS_SESSION = uuid.UUID("f62c48dfdd33416a98121ff4b004817a")
"""Namespace for session ids."""
NS_GRAPH = uuid.UUID("15f123374f9f4a9385a5f31b6ed3f630")
"""Namespace for graph ids."""
NS_NODE = uuid.UUID("3446718135514d6c9ea36b5377ea0a19")
"""Namespace for node ids."""
NS_QUEUE = uuid.UUID("55fbaa9565fa4bf3ba806ade03c1e395")
"""Namespace for queue ids."""
UUID_OUTPUT_QUEUE = uuid.UUID("8a82aef7da2c4b49bf7ba25d42e99391")
"""Output queue id."""
TEST_EXECUTOR = uuid.UUID("5883f8c8a5bc4d56ae3202d1dc548118")
"""Executor UUID for tests."""


DEBUG_OUTPUT_LENGTH: int | None = None
"""The length of the id shown when debugging. The id is truncated to this
length for easier readability. If it's set to None the full id is shown."""


def set_debug_output_length(output_length: int | None) -> None:
    """
    Sets the length of ids when shown in debug outputs.

    Args:
        output_length (int | None): The maximum display length. If None the
            full id is shown.
    """
    global DEBUG_OUTPUT_LENGTH  # pylint: disable=global-statement

    DEBUG_OUTPUT_LENGTH = output_length


def trim_id(text_id: str) -> str:
    """
    Prepares the text representation of an id for user output. The id might
    get truncated based on the debug settings.

    Args:
        text_id (str): The id representation to prepare.

    Returns:
        str: The potentially truncated id representation.
    """
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
        Use `parse` or the respective `create` method instead.

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

    @staticmethod
    def length() -> int:
        """
        Get the length of the id in characters.

        Returns:
            int: The length if the id is represented as string.
        """
        return 33

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
        if len(text) != cls.length() or "-" in text:
            raise ValueError(f"invalid {cls.__name__}: {text}")
        return cls(uuid.UUID(text[1:]))

    @classmethod
    def parse_folder(cls: type[SelfT], path: str) -> SelfT | None:
        """
        Parses a folder path as id. This is done backwards from the leaf
        folder.

        Args:
            cls (type[SelfT]): The id class.
            path (str): The path.

        Returns:
            SelfT | None: The id or None if the path did not represent an id.
        """
        full = ""
        length = cls.length()
        while path:
            path, segment = os.path.split(path)
            if not segment:
                continue
            full = f"{full}{segment}"
            if len(full) >= length:
                break
        try:
            return cls.parse(full)
        except ValueError:
            return None

    @classmethod
    def find_folder_ids(cls: type[SelfT], path: str) -> Iterable[SelfT]:
        """
        Scans a given folder for subpaths that represent ids.

        Args:
            cls (type[SelfT]): The id class.
            path (str): The root path (not part of any id).

        Yields:
            SelfT: The ids.
        """
        length = cls.length()

        def scan_folder(prefix: str, folder: str) -> Iterable[SelfT]:
            for inner in get_subfolders(folder):
                candidate = f"{prefix}{inner}"
                if len(candidate) >= length:
                    try:
                        yield cls.parse(candidate)
                    except ValueError:
                        pass
                else:
                    yield from scan_folder(
                        candidate, os.path.join(folder, inner))

        yield from scan_folder("", path)

    def _hex(self) -> str:
        """
        Returns the raw hex value of the id.

        Returns:
            str: The raw hex value.
        """
        return self._id.hex

    def to_parseable(self) -> str:
        """
        Creates a parseable representation of the id.

        Returns:
            str: A string that can be parsed by `parse`.
        """
        return f"{self.prefix()}{self._hex()}"

    def as_folder(self) -> list[str]:
        """
        Creates a representation that can be used as folder structure.

        Returns:
            list[str]: Each element is a path segment.
        """
        out = self.to_parseable()
        return [out[:3], out[3:]]

    def to_tensor(self) -> torch.Tensor:
        """
        Convert the id to a tensor value.

        Returns:
            torch.Tensor: The tensor representing the id. Use `parse_tensor` to
                parse.
        """
        prefix = list(self.prefix().encode("utf-8"))
        bid = list(self._id.bytes)
        return create_tensor(prefix + bid, dtype="uint8")

    @staticmethod
    def tensor_shape() -> list[int]:
        """
        The shape of the tensor representation.

        Returns:
            list[int]: The shape is 1 prefix byte and 16 uuid bytes.
        """
        return [17]

    @classmethod
    def parse_tensor(cls: type[SelfT], val: torch.Tensor) -> SelfT:
        """
        Parse an id from a tensor.

        Args:
            cls (type[SelfT]): The id class.
            val (torch.Tensor): The parseable tensor.

        Raises:
            ValueError: If the tensor cannot be interpreted as id.

        Returns:
            SelfT: The id.
        """
        val = val.ravel()
        if list(val.shape) != cls.tensor_shape():
            raise ValueError(
                f"invalid shape for tensor {list(val.shape)} != "
                f"{cls.tensor_shape()}")
        prefix = tensor_to_str(val[0])
        if prefix != cls.prefix():
            raise ValueError(f"invalid prefix for {cls.__name__}: {prefix}")
        ubytes = bytes(tensor_list(val[1:]))  # type: ignore
        return cls(uuid.UUID(bytes=ubytes))

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


class UserId(BaseId):
    """A `UserId` is used to identify a user."""
    @staticmethod
    def prefix() -> str:
        return "U"

    @staticmethod
    def create(uname: UName) -> 'UserId':
        """
        Creates a `UserId` from a given user name.

        Args:
            uname (UName): The user name.

        Returns:
            UserId: The corrseponding user id.
        """
        return UserId(uuid.uuid5(NS_USER, uname.get()))


class SessionId(BaseId):
    """A `SessionId` is used to identify a session of a user."""
    @staticmethod
    def prefix() -> str:
        return "S"

    @staticmethod
    def create_for(user_id: UserId) -> 'SessionId':
        """
        Create a new unique `SessionId` for a given user.

        Args:
            user_id (UserId): The user id.

        Returns:
            SessionId: The unique session id.
        """
        return SessionId(uuid.uuid5(
            NS_SESSION,
            f"{user_id.to_parseable()}{NAME_SEP}{uuid.uuid4().hex}"))


class GraphId(BaseId):
    """A `GraphId` is used to identify an execution graph."""
    @staticmethod
    def prefix() -> str:
        return "G"

    @staticmethod
    def create(gname: QualifiedGraphName) -> 'GraphId':
        """
        Creates a `GraphId` from a given graph name.

        Args:
            gname (QualifiedGraphName): The qualified graph name.

        Returns:
            GraphId: The corrseponding graph id.
        """
        return GraphId(uuid.uuid5(NS_GRAPH, gname.to_parseable()))


class NodeId(BaseId):
    """
    A `NodeId` is used to identify a node in the execution graph.
    """
    @staticmethod
    def prefix() -> str:
        return "N"

    @staticmethod
    def create(gname: QualifiedGraphName, name: NName) -> 'NodeId':
        """
        Creates a `NodeId` from a given node name.

        Args:
            gname (QualifiedGraphName): The qualified graph name.
            name (NName): The node name.

        Returns:
            NodeId: The corrseponding node id.
        """
        return NodeId(uuid.uuid5(
            NS_NODE, f"{gname.to_parseable()}{NAME_SEP}{name.get()}"))


class QueueId(BaseId):
    """
    A `QueueId` is used to identify a queue in the execution graph.
    """
    @staticmethod
    def prefix() -> str:
        return "Q"

    @staticmethod
    def create(gname: QualifiedGraphName, node_name: NName) -> 'QueueId':
        """
        Creates a `QueueId` for a given graph name and node name.

        Args:
            gname (QualifiedGraphName): The qualified graph name.
            node_name (NName): The node name.

        Returns:
            QueueId: The corrseponding queue id.
        """
        return QueueId(uuid.uuid5(
            NS_QUEUE, f"{gname.to_parseable()}{NAME_SEP}{node_name.get()}"))

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
        """
        The executor id used for test cases.

        Returns:
            ExecutorId: The test executor id.
        """
        return TEST_EXECUTOR_ID


TEST_EXECUTOR_ID = ExecutorId(TEST_EXECUTOR)
"""The executor id used for test cases."""


def once_test_executors(execs: list[ExecutorId]) -> Callable[[], ExecutorId]:
    """
    Creates an executor id generator from a list of executor ids. Each entry
    in the list gets returned exactly once. If the list is exhausted an error
    is raised.

    Args:
        execs (list[ExecutorId]): The list of executor ids to return by the
            generator.

    Raises:
        ValueError: If the generator is exhausted.

    Returns:
        Callable[[], ExecutorId]: Returns an executor id.
    """
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
    A data id is used to identify a piece of data in temporary storage
    (payload data). The exact format of a data id is left to the storage to
    decide.
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


Locality = Literal[
    "local",
    "remote",
    "either",
]
"""In which environment a module can be used. It can be 'local' only, 'remote'
only, or available in 'either' of those environments."""

L_LOCAL: Locality = "local"
"""Indicates that a module can only be used in a local environment."""
L_REMOTE: Locality = "remote"
"""Indicates that a module can only be used in a remote environment."""
L_EITHER: Locality = "either"
"""
Indicates that a module can be used in either a local or remote environment.
"""


class Module:  # pylint: disable=too-few-public-methods
    """
    A module for environment dependent behavior. Module classes need to be
    subclassed to implement the respective behavior.
    """
    def locality(self) -> Locality:
        """
        Whether the module is for a local (same process) environment only, a
        possibly remote (other process / other node) environment, or either.
        All loaded modules must be either all local or all remote (not counting
        modules that indicate either). A module should only be L_EITHER if it
        serves as caching layer or it behaves the same in a local and remote
        context (i.e., no data is stored in the current process and data is
        possible to be accessed remotely)

        Returns:
            Locality: The locality mode.
        """
        raise NotImplementedError()


class CacheId:
    """A cache id identifies a cacheable input for a given graph."""
    def __init__(self, graph_id: GraphId, cache_id: str) -> None:
        """
        Creates a cache id.

        Args:
            graph_id (GraphId): The graph.
            cache_id (str): The raw hash to identify the input.
        """
        if NAME_SEP in cache_id:
            raise ValueError(f"invalid cache id: {cache_id}")
        self._graph_id = graph_id
        self._cache_id = cache_id

    def get_graph_id(self) -> GraphId:
        """
        Gets the graph.

        Returns:
            GraphId: The graph.
        """
        return self._graph_id

    def get_cache_id(self) -> str:
        """
        Returns the raw input hash.

        Returns:
            str: The raw input hash.
        """
        return self._cache_id

    def to_parseable(self) -> str:
        """
        Convert the cache id into a parseable representation.

        Returns:
            str: The cache id string.
        """
        return (
            f"{self.get_graph_id().to_parseable()}"
            f"{NAME_SEP}{self.get_cache_id()}")

    @staticmethod
    def parse(text: str) -> 'CacheId':
        """
        Parses a cache id.

        Args:
            text (str): The cache id string.

        Returns:
            CacheId: The parsed cache id.
        """
        graph_id_str, cache_id_str = text.split(NAME_SEP, 1)
        return CacheId(GraphId.parse(graph_id_str), cache_id_str)

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, CacheId):
            return False
        if self.get_graph_id() != other.get_graph_id():
            return False
        return self.get_cache_id() == other.get_cache_id()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._cache_id)

    def __str__(self) -> str:
        return f"gid={self.get_graph_id()},cid={self.get_cache_id()}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.__str__()}]"
