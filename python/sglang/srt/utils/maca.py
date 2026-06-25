import os
from pathlib import Path
from typing import Mapping

import torch


def get_torch_maca_version(torch_module=torch) -> str | None:
    return getattr(torch_module.version, "maca", None)


def has_maca_toolkit_env(env: Mapping[str, str] | None = None) -> bool:
    env = env or os.environ
    maca_path = env.get("MACA_PATH")
    if not maca_path:
        return False
    maca_root = Path(maca_path)
    return (
        (maca_root / "tools" / "cu-bridge").exists()
        or (maca_root / "mxgpu_llvm").exists()
        or (maca_root / "lib").exists()
    )


def is_maca_available(torch_module=torch, env: Mapping[str, str] | None = None) -> bool:
    if get_torch_maca_version(torch_module) is not None:
        return True
    if not has_maca_toolkit_env(env):
        return False
    cuda = getattr(torch_module, "cuda", None)
    return bool(cuda is not None and cuda.is_available())
