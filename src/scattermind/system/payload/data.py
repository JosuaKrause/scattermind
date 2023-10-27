from typing import TypeVar

import torch

from scattermind.system.base import DataId, Module
from scattermind.system.info import DataInfo
from scattermind.system.torch_util import deserialize_tensor, serialize_tensor


DT = TypeVar('DT', bound=DataId)


class DataStore(Module):
    def store_tensor(self, value: torch.Tensor) -> DataId:
        return self.store_data(serialize_tensor(value))

    def get_tensor(
            self,
            data_id: DataId,
            data_info: DataInfo) -> torch.Tensor | None:
        data = self.get_data(data_id)
        if data is None:
            return None
        res = deserialize_tensor(data, data_info.dtype())
        return data_info.check_tensor(res)

    def ensure_id_type(self, data_id: DataId, data_id_type: type[DT]) -> DT:
        if not isinstance(data_id, data_id_type):
            raise ValueError(
                f"unexpected {data_id.__class__.__name__}: {data_id}")
        return data_id

    @staticmethod
    def is_content_addressable() -> bool:
        raise NotImplementedError()

    def store_data(self, data: bytes) -> DataId:
        raise NotImplementedError()

    def get_data(self, data_id: DataId) -> bytes | None:
        raise NotImplementedError()

    def data_id_type(self) -> type[DataId]:
        raise NotImplementedError()
