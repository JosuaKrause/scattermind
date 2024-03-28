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
"""Defines the caching interface for caching graph input and outputs."""
import hashlib

from scattermind.system.base import CacheId, GraphId, Module
from scattermind.system.info import DataFormat
from scattermind.system.payload.values import TaskValueContainer
from scattermind.system.redis_util import tensor_to_redis


class GraphCache(Module):
    """A caching layer for graph input and outputs."""
    def get_cache_id(
            self,
            graph_id: GraphId,
            input_format: DataFormat,
            tvc: TaskValueContainer) -> CacheId:
        """
        Computes the cache id for the given input.

        Args:
            graph_id (GraphId): The graph id.
            input_format (DataFormat): The format of the input.
            tvc (TaskValueContainer): The data.

        Returns:
            CacheId: The cache id.
        """
        blake = hashlib.blake2b(digest_size=32)
        for key in sorted(input_format.keys()):
            key_bytes = key.encode("utf-8")
            blake.update(f"{len(key_bytes)}:".encode("utf-8"))
            blake.update(key_bytes)
            # NOTE: we cannot compress the data as it would yield different
            # results almost every time
            value_bytes = tensor_to_redis(
                tvc[key], compress=False).encode("utf-8")
            blake.update(f"{len(value_bytes)}:".encode("utf-8"))
            blake.update(value_bytes)
        return CacheId(graph_id, blake.hexdigest())

    def put_cached_output(
            self,
            cache_id: CacheId,
            output_data: TaskValueContainer) -> None:
        """
        Caches the given result.

        Args:
            cache_id (CacheId): The cache id.
            output_data (TaskValueContainer): The data.
        """
        raise NotImplementedError()

    def get_cached_output(
            self,
            cache_id: CacheId,
            output_format: DataFormat) -> TaskValueContainer | None:
        """
        Retrieves the cached data.

        Args:
            cache_id (CacheId): The cache id.
            output_format (DataFormat): The expected output format.

        Returns:
            TaskValueContainer | None: The data if it is available.
        """
        raise NotImplementedError()
