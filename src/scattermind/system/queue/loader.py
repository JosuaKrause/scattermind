from typing import Literal, TypedDict

from typing_extensions import NotRequired

from scattermind.system.plugins import load_plugin
from scattermind.system.queue.queue import QueuePool


LocalQueuePoolModule = TypedDict('LocalQueuePoolModule', {
    "name": Literal["local"],
    "check_assertions": NotRequired[bool],
})


QueuePoolModule = LocalQueuePoolModule


def load_queue_pool(module: QueuePoolModule) -> QueuePool:
    if "." in module["name"]:
        kwargs = dict(module)
        plugin = load_plugin(QueuePool, f"{kwargs.pop('name')}")
        return plugin(**kwargs)
    if module["name"] == "local":
        from scattermind.system.queue.local import LocalQueuePool
        return LocalQueuePool(
            check_assertions=module.get("check_assertions", False))
    raise ValueError(f"unknown queue pool: {module['name']}")
