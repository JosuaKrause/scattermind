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
"""A caching layer that does not do any caching."""
from scattermind.system.base import CacheId, L_EITHER, Locality
from scattermind.system.cache.cache import GraphCache
from scattermind.system.info import DataFormat
from scattermind.system.payload.values import TaskValueContainer


class NoCache(GraphCache):
    """No caching is performed with this graph cache."""
    @staticmethod
    def locality() -> Locality:
        return L_EITHER

    def put_cached_output(
            self,
            cache_id: CacheId,
            output_data: TaskValueContainer) -> None:
        pass

    def get_cached_output(
            self,
            cache_id: CacheId,
            output_format: DataFormat) -> TaskValueContainer | None:
        return None
