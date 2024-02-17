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
"""This module defines name classes. The classes guarantee that no invalid
characters are used in a name. Names are comparable for equality and can be
used as keys in dictionaries."""
NAME_SEP = ":"
"""Separator for names and queue id generation."""
INVALID_CHARACTERS = {
    " ",
    "\n",
    "\r",
    "\t",
    "\0",
    "\\",
    "\'",
    "\"",
    "\a",
    "\b",
    "\f",
    "\v",
}
"""Characters specifically forbidden as characters in names."""


class NameStr:
    """A class to define a general name."""
    def __init__(self, name: str) -> None:
        """
        Creates a name from the given string.

        Args:
            name (str): The raw string.

        Raises:
            ValueError: If the name is not valid.
        """
        if not self.is_valid_name(name):
            raise ValueError(f"name {name} is not valid")
        self._name = name

    def get(self) -> str:
        """
        Gets the raw name string.

        Returns:
            str: The name as string.
        """
        return self._name

    @staticmethod
    def is_valid_name(name: str) -> bool:
        """
        Verifies that the raw name string is valid.

        Args:
            name (str): The raw name string.

        Returns:
            bool: Whether the string represents a valid name.
        """
        return (
            len(name) > 0
            and NAME_SEP not in name
            and not set(name).intersection(INVALID_CHARACTERS)
        )

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, self.__class__):
            return False
        return self._name == other._name

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._name)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self._name}]"

    def __repr__(self) -> str:
        return self.__str__()


class NName(NameStr):
    """Name of a execution node. Node names are deterministically converted
    into node ids internally."""


class QName(NameStr):
    """Name of a queue. Queue names are deterministically converted into queue
    ids internally."""


class GName(NameStr):
    """Name of a graph. Graph names are deterministically converted into graph
    ids internally."""


class GNamespace(NameStr):
    """Namespace of a set of graphs."""


class QualifiedGraphName:
    """A fully specified graph name. It consists of a namespace and graph name.
    """
    def __init__(self, namespace: GNamespace, name: GName) -> None:
        """
        Creates a qualified graph name.

        Args:
            namespace (GNamespace): The namespace.
            name (GName): The graph name.
        """
        self._namespace = namespace
        self._name = name

    def get_namespace(self) -> GNamespace:
        """
        Gets the namespace.

        Returns:
            GNamespace: The namespace.
        """
        return self._namespace

    def get_name(self) -> GName:
        """
        Gets the graph name.

        Returns:
            GName: The graph name.
        """
        return self._name

    def to_parseable(self) -> str:
        """
        Retrieves an unambiguous string representation.

        Returns:
            str: The qualified graph name as string.
        """
        return f"{self._namespace.get()}{NAME_SEP}{self._name.get()}"

    @staticmethod
    def parse(text: str) -> 'QualifiedGraphName':
        """
        Parses a string into a qualified graph name. The format is expected to
        be '{namespace}:{graph name}'.

        Args:
            text (str): The string to parse.

        Raises:
            ValueError: If the input is invalid.

        Returns:
            QualifiedGraphName: The qualified graph name.
        """
        if NAME_SEP not in text:
            raise ValueError(
                f"{text} is not a qualified graph name: 'ns:graph'")
        ns, gname = text.split(NAME_SEP, 1)
        return QualifiedGraphName(GNamespace(ns), GName(gname))

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, self.__class__):
            return False
        return (
            self._namespace == other._namespace and self._name == other._name)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self._namespace, self._namespace))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self.to_parseable()}]"

    def __repr__(self) -> str:
        return self.__str__()


class QualifiedName:
    """A qualified name describes a value field and how to access it. It
    combines a node name and a value field name. The node name can be None
    to describe input value fields. In the execution graph json definition
    qualified names are the node name separated by ':' from the value field
    name as defined in the node.

    Examples: 'node_2:out' addresses the output value field 'out' from the
    node named 'node_2'. ':text' addresses the input value field 'text' of
    the current execution graph."""
    def __init__(self, nname: NName | None, vname: str) -> None:
        """
        Creates a qualified name from its individual parts.

        Args:
            nname (NName | None): The node name or None to specify the input
                of the current execution graph.
            vname (str): The value field name of the node or execution graph.

        Raises:
            ValueError: If the value field name is not valid.
        """
        if not self.is_valid_name(vname):
            raise ValueError(f"value name {vname} is not valid")
        self._nname = nname
        self._vname = vname

    @staticmethod
    def is_valid_name(vname: str) -> bool:
        """
        Whether the given value field name is valid.

        Args:
            vname (str): The raw value field name string.

        Returns:
            bool: Whether the name is valid.
        """
        return (
            len(vname) > 0
            and NAME_SEP not in vname
            and not set(vname).intersection(INVALID_CHARACTERS)
        )

    def get_value_name(self) -> str:
        """
        Get the raw value field name string.

        Returns:
            str: The raw string.
        """
        return self._vname

    def to_parseable(self) -> str:
        """
        Get a parseable representation of the qualified name as it would be
        found in the execution graph json.

        Examples: 'node_2:out' addresses the output value field 'out' from the
        node named 'node_2'. ':text' addresses the input value field 'text' of
        the current execution graph (the node name is None in this case).

        Returns:
            str: The qualified name.
        """
        if self._nname is None:
            return f"{NAME_SEP}{self._vname}"
        return f"{self._nname.get()}{NAME_SEP}{self._vname}"

    @staticmethod
    def parse(text: str) -> 'QualifiedName':
        """
        Parses a string into a qualified name. The format is expected to be
        '{node name}:{value field name}' or ':{value field name}' for the
        input of the current execution graph.

        Args:
            text (str): The string to parse.

        Raises:
            ValueError: If the input is invalid.

        Returns:
            QualifiedName: The qualified name.
        """
        if NAME_SEP not in text:
            raise ValueError(
                f"{text} is not a qualified name: 'node:value'")
        nname_str, vname = text.split(NAME_SEP, 1)
        if not nname_str:
            return QualifiedName(None, vname)
        return QualifiedName(NName(nname_str), vname)

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, self.__class__):
            return False
        return self._nname == other._nname and self._vname == other._vname

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self._nname, self._vname))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self.to_parseable()}]"

    def __repr__(self) -> str:
        return self.__str__()


ValueMap = dict[str, QualifiedName]
"""A map of input value field names to output qualified names. This is used
to connect inputs of a node to the correct outputs of previous nodes or an
input of the current execution graph."""
