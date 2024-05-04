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
from collections.abc import Iterable

from scattermind.system.base import Module, SessionId, UserId


class Session:
    def __init__(
            self,
            sessions: 'SessionStore',
            session_id: SessionId) -> None:
        self._sid = session_id
        self._sessions = sessions

    def get_session_id(self) -> SessionId:
        return self._sid

    def remove(self) -> None:
        self._sessions.remove(self._sid)


class SessionStore(Module):
    def __init__(self) -> None:
        pass

    def get_sessions(self, user_id: UserId) -> Iterable[Session]:
        raise NotImplementedError()

    def get_session(self, session_id: SessionId) -> Session:
        return Session(self, session_id)

    def remove(self, session_id: SessionId) -> None:
        raise NotImplementedError()
