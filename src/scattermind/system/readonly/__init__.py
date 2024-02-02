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
"""Provides a unified interface for nodes to access external data.
While nodes could, in theory, access data via normal file system operations,
this would not be very portable and wouldn't provide a good user experience.
Through the `readonly` interfaces a node can access data through a unspecified
key value store of blobs. The blobs can be read via random access. The data is
meant to be unchangeable (hence `readonly`) and is most commonly used for
stored weights."""
