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
from scattermind.system.base import GraphId, L_EITHER, Locality
from scattermind.system.cache.cache import GraphCache
from scattermind.system.info import DataFormat
from scattermind.system.payload.values import ComputeValues, LazyValues


class NoCache(GraphCache):
    @staticmethod
    def locality() -> Locality:
        return L_EITHER

    def put_cached_output(
            self,
            graph_id: GraphId,
            input_format: DataFormat,
            input_data: dict[str, LazyValues],
            output_format: DataFormat,
            output_data: dict[str, LazyValues]) -> None:
        pass

    def get_cached_output(
            self,
            graph_id: GraphId,
            input_format: DataFormat,
            input_data: dict[str, LazyValues]) -> ComputeValues | None:
        return None
