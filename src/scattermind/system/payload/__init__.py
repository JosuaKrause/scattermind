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
