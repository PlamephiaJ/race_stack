"""Require CUDA ray casting and select the injected Jetson driver before CUDA init."""

import ctypes
from pathlib import Path
import platform


_driver = None


def require_cuda():
    global _driver
    import range_libc

    if not getattr(range_libc, 'SHOULD_USE_CUDA', False):
        raise RuntimeError(
            'PF requires CUDA RangeLibc (SHOULD_USE_CUDA=True). Run '
            '.install_utils/particle_filter_setup.sh; CPU fallback is disabled.')

    # Some Jetson CUDA images put an old compat driver before the injected host
    # driver. Load the host SONAME explicitly before libcudart opens libcuda.
    host_driver = Path('/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1')
    driver_path = (str(host_driver) if platform.machine() == 'aarch64'
                   and host_driver.is_file() else 'libcuda.so.1')
    try:
        _driver = ctypes.CDLL(driver_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        raise RuntimeError(f'PF cannot load CUDA driver {driver_path}: {error}') from error
    status = _driver.cuInit(0)
    if status:
        raise RuntimeError(
            f'PF CUDA driver initialization failed (CUDA error {status}). '
            'On Jetson check device access to /dev/nvmap, /dev/nvhost-*gpu and /dev/dri/renderD*. '
            'CPU fallback is disabled.')
    devices = ctypes.c_int()
    status = _driver.cuDeviceGetCount(ctypes.byref(devices))
    if status or devices.value < 1:
        raise RuntimeError(
            f'PF needs an accessible CUDA GPU (error={status}, devices={devices.value}); '
            'CPU fallback is disabled.')
    return devices.value
