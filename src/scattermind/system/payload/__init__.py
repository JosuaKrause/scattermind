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
"""The payload module handles interfaces for handling data that gets produced
during graph execution. Unlike the original input data, data from intermediate
and final results, aka payload data, is stored in volatile storage. This allows
for graceful handling of overload. The implementations are free to choose under
which circumstances payload data will be freed (e.g., after a certain amount of
time or when running low on memory). Every consumer of payload data must handle
the case where data is no longer available. By default, if payload data is
missing for a task, the task should be requeued anew (using the non-volatile
input data) and the retries counter should be increased by one. If the retry
counter grows above a certain threshold an error should be emitted instead."""
