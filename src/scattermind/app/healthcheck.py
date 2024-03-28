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
"""Provides an API based worker healthcheck."""
import threading
from typing import TypedDict

import requests
from quick_server import create_server, QuickServer
from quick_server import QuickServerRequestHandler as QSRH
from quick_server import ReqArgs

from scattermind.api.api import ScattermindAPI
from scattermind.api.loader import VersionInfo
from scattermind.system.config.config import Config
from scattermind.system.executor.executor import ExecutorManager
from scattermind.system.util import get_time_str


InfoResponse = TypedDict('InfoResponse', {
    "version": str,
    "start_date": str,
    "executors": int,
    "app_version": str | None,
    "app_commit": str | None,
    "deploy_date": str | None,
})


def init_healthcheck(
        addr: str,
        port: int,
        executor_manager: ExecutorManager,
        version_info: VersionInfo | None) -> tuple[QuickServer, str]:
    """
    Initializes the healthcheck API.

    Args:
        addr (str): The address to serve.
        port (int): The port to serve.
        executor_manager (ExecutorManager): The executor manager.
        version_info (VersionInfo | None): External version info.

    Returns:
        tuple[QuickServer, str]: The server and prefix tuple.
    """
    import scattermind  # pylint: disable=import-outside-toplevel

    server: QuickServer = create_server(
        (addr, port),
        parallel=True,
        thread_factory=threading.Thread,
        token_handler=None,
        worker_constructor=None,
        soft_worker_death=True)

    prefix = "/api"

    server.suppress_noise = False

    def report_slow_requests(
            method_str: str, path: str, duration: float) -> None:
        print(f"slow request {method_str} {path} ({duration}s)")

    server.report_slow_requests = report_slow_requests

    server_timeout = 10 * 60
    server.timeout = server_timeout
    server.socket.settimeout(server_timeout)

    server.no_command_loop = True

    start_date = get_time_str()
    version = f"{scattermind.__name__}/{scattermind.__version__}"

    server.update_version_string(version)

    server.set_common_invalid_paths(["/", "//"])

    if version_info is None:
        app_version = None
        app_commit = None
        deploy_date = None
    else:
        app_version, app_commit, deploy_date = version_info

    @server.json_get(f"{prefix}/info")
    def _get_info(_req: QSRH, _rargs: ReqArgs) -> InfoResponse:
        return {
            "version": version,
            "start_date": start_date,
            "executors": executor_manager.active_count(),
            "app_version": app_version,
            "app_commit": app_commit,
            "deploy_date": deploy_date,
        }

    return server, prefix


HEALTHCHECK_SERVER: QuickServer | None = None
"""The healthcheck server."""


def start_healthcheck(server: QuickServer, prefix: str) -> None:
    """
    Starts the healthcheck API server.

    Args:
        server (QuickServer): The server.
        prefix (str): The URL prefix.
    """
    global HEALTHCHECK_SERVER  # pylint: disable=global-statement

    HEALTHCHECK_SERVER = server

    def start() -> None:
        addr, port = server.server_address
        if not isinstance(addr, str):
            addr = addr.decode("utf-8")
        print(f"starting healthcheck at http://{addr}:{port}{prefix}/")
        try:
            server.serve_forever()
        finally:
            print("shutting down healthcheck..")
            server.server_close()

    th = threading.Thread(target=start, daemon=True)
    th.start()


def stop_healthcheck() -> None:
    """Stops the healthcheck if it was running."""
    global HEALTHCHECK_SERVER  # pylint: disable=global-statement

    server = HEALTHCHECK_SERVER
    HEALTHCHECK_SERVER = None
    if server is not None:
        server.done = True


def maybe_start_healthcheck(
        config: Config, version_info: VersionInfo | None) -> None:
    """
    Starts the healthcheck API if it was configured.

    Args:
        config (Config): The config.
        version_info (VersionInfo | None): External version info.
    """
    hc = config.get_healthcheck()
    if hc is None:
        return
    _, addr, port = hc
    server, prefix = init_healthcheck(
        addr, port, config.get_executor_manager(), version_info)
    start_healthcheck(server, prefix)


def perform_healthcheck(config: ScattermindAPI) -> int:
    """
    Performs the healthcheck by contacting the API.

    Args:
        config (Config): The config.

    Returns:
        int: The number of active executors. If this number is 0 then the
            healthcheck failed.
    """
    hc = config.get_healthcheck()
    if hc is None:
        return 0
    addr, _, port = hc
    resp = requests.get(f"http://{addr}:{port}/api/info", timeout=60)
    resp.raise_for_status()
    obj = resp.json()
    return int(obj["executors"])
