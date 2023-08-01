# Copyright The Lightning AI team.
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

from typing import List, Union, Optional
from functools import lru_cache
import torch

_has_directml = False
try:
    import torch_directml
    _has_directml = True
except:
    pass

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry


class DMLAccelerator(Accelerator):
    """Accelerator for DML devices."""

    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not DML.
        """
        if device.type != "privateuseone":
            raise ValueError(f"Device should be DML, got {device} instead.")

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        if not _has_directml:
            return []
        from lightning.fabric.utilities.device_parser import _parse_gpu_ids
        return _parse_gpu_ids(devices, include_dml=True)

    @staticmethod
    def get_parallel_devices(devices: Union[int, str, List[int]]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        if not _has_directml:
            return []
        parsed_devices = DMLAccelerator.parse_devices(devices)
        assert parsed_devices is not None
        return [torch_directml.device(i) for i in parsed_devices]

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        if not _has_directml:
            return 0
        return torch_directml.device_count()

    @staticmethod
    @lru_cache(1)
    def is_available() -> bool:
        """DML might not always be available for execution."""
        if not _has_directml:
            return False
        return torch_directml.device_count() > 0

    @classmethod
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        if _has_directml:
            accelerator_registry.register(
                "dml",
                cls,
                description=cls.__class__.__name__,
            )

def _get_all_available_dml_gpus() -> List[int]:
    """
    Returns:
        A list of all available DML GPUs
    """
    if not _has_directml:
        return []
    return [i for i in range(torch_directml.device_count())]

# DMLPatch
