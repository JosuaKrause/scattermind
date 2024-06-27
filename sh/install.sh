#!/usr/bin/env bash
#
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
#
set -ex

cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )/../" &> /dev/null

PYTHON="${PYTHON:-python}"
which ${PYTHON} > /dev/null
if [ $? -ne 0 ]; then
    PYTHON=python
fi

MAJOR=$(${PYTHON} -c 'import sys; print(sys.version_info.major)')
MINOR=$(${PYTHON} -c 'import sys; print(sys.version_info.minor)')
echo "${PYTHON} v${MAJOR}.${MINOR}"
if [ ${MAJOR} -eq 3 ] && [ ${MINOR} -lt 11 ] || [ ${MAJOR} -lt 3 ]; then
    echo "${PYTHON} version must at least be 3.11" >&2
    exit 1
fi

! read -r -d '' PY_TORCH_VERIFY <<'EOF'
import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

print(f"backend is (cpu|cuda|mps): {get_device()}")
EOF

PYTORCH=
if ${PYTHON} -c 'import torch;assert torch.__version__.startswith("2.")' &>/dev/null 2>&1; then
    PYTORCH=$(${PYTHON} -c 'import torch;print(torch.__version__)')
else
    if [ ! $CI = "true" ] && command -v conda &>/dev/null 2>&1; then
        conda install -y pytorch torchvision torchaudio -c pytorch
    else
        ${PYTHON} -m pip install --progress-bar off torch torchvision torchaudio
    fi
fi

${PYTHON} -m pip install --progress-bar off --upgrade pip
${PYTHON} -m pip install --progress-bar off --upgrade -r requirements.txt
${PYTHON} -m pip install --progress-bar off --upgrade -r requirements.dev.txt

# change branches here when using a development branch for redipy or
# quick-server change to main and master to deactivate
REDIPY_BRANCH="main"
QS_BRANCH="master"

if [ "${REDIPY_BRANCH}" != "main" ] && [ ! -z "${USE_DEV}" ]; then
    REDIPY_PATH="../redipy"
    REDIPY_URL="git+https://github.com/JosuaKrause/redipy.git"
    ${PYTHON} -m pip uninstall -y redipy
    if [ -d "${REDIPY_PATH}" ]; then
        ${PYTHON} -m pip install --upgrade -e "${REDIPY_PATH}"
    else
        ${PYTHON} -m pip install --upgrade "${REDIPY_URL}@${REDIPY_BRANCH}"
    fi
fi

if [ "${QS_BRANCH}" != "master" ] && [ ! -z "${USE_DEV}" ]; then
    QS_PATH="../quick_server"
    QS_URL="git+https://github.com/JosuaKrause/quick_server.git"
    ${PYTHON} -m pip uninstall -y quick-server
    if [ -d "${QS_PATH}" ]; then
        ${PYTHON} -m pip install --upgrade -e "${QS_PATH}"
    else
        ${PYTHON} -m pip install --upgrade "${QS_URL}@${QS_BRANCH}"
    fi
fi

if [ ! -z "${PYTORCH}" ]; then
    ${PYTHON} -c "${PY_TORCH_VERIFY}"
    echo "pytorch available: ${PYTORCH}"
else
    echo "installed pytorch. it's probably better if you install it yourself"
    echo "for MacOS follow these instructions: https://developer.apple.com/metal/pytorch/"
fi
