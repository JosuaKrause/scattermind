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


class NameStr:
    def __init__(self, name: str) -> None:
        if not self.is_valid_name(name):
            raise ValueError(f"name {name} is not valid")
        self._name = name

    def get(self) -> str:
        return self._name

    @staticmethod
    def is_valid_name(name: str) -> bool:
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
    pass


class QName(NameStr):
    pass


class GName(NameStr):
    pass


class QualifiedName:
    def __init__(self, nname: NName | None, vname: str) -> None:
        if not self.is_valid_name(vname):
            raise ValueError(f"value name {vname} is not valid")
        self._nname = nname
        self._vname = vname

    @staticmethod
    def is_valid_name(vname: str) -> bool:
        return (
            len(vname) > 0
            and NAME_SEP not in vname
            and not set(vname).intersection(INVALID_CHARACTERS)
        )

    def get_value_name(self) -> str:
        return self._vname

    def to_parseable(self) -> str:
        if self._nname is None:
            return f"{NAME_SEP}{self._vname}"
        return f"{self._nname.get()}{NAME_SEP}{self._vname}"

    @staticmethod
    def parse(text: str) -> 'QualifiedName':
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
