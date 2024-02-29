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
"""Provides a unified interface for nodes to access external data.
While nodes could, in theory, access data via normal file system operations,
this would not be very portable and wouldn't provide a good user experience.
Through the `readonly` interfaces a node can access data through a unspecified
key value store of blobs. The blobs can be read via random access. The data is
meant to be unchangeable (hence `readonly`) and is most commonly used for
stored weights."""
