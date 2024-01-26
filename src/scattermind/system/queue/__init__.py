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
"""Queues contain tasks waiting for execution in a node. When executing, nodes
take tasks from their input queue, execute, and then put the task in one of
their output queues. The special `OUTPUT_QUEUE` is the destination of all
completed tasks, while regular output queues are input queues of other nodes.
There are implementable strategies to decide which queue to pick for processing
(and in turn which node to execute) and which and how many tasks are taken from
a queue when the node executes."""
