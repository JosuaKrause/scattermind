import torch

from scattermind.system.base import DataId, Module
from scattermind.system.info import DataInfo
from scattermind.system.torch_util import deserialize_tensor, serialize_tensor


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

    @staticmethod
    def is_content_addressable() -> bool:
        raise NotImplementedError()

    def store_data(self, data: bytes) -> DataId:
        raise NotImplementedError()

    def get_data(self, data_id: DataId) -> bytes | None:
        raise NotImplementedError()
