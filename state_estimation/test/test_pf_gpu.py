"""GPU configuration must fail clearly rather than use a CPU fallback."""
from types import SimpleNamespace
import sys

import pytest
from state_estimation import particle_filter_gpu as gpu


def fake_driver(monkeypatch, *, cuda=True, init=0, count=1):
    monkeypatch.setitem(sys.modules, 'range_libc', SimpleNamespace(SHOULD_USE_CUDA=cuda))
    monkeypatch.setattr(gpu.platform, 'machine', lambda: 'x86_64')

    def get_count(pointer):
        pointer._obj.value = count
        return 0

    driver = SimpleNamespace(cuInit=lambda flags: init, cuDeviceGetCount=get_count)
    monkeypatch.setattr(gpu.ctypes, 'CDLL', lambda *a, **kw: driver)


def test_cpu_only_extension_is_rejected(monkeypatch):
    fake_driver(monkeypatch, cuda=False)
    with pytest.raises(RuntimeError, match='CPU fallback is disabled'):
        gpu.require_cuda()


def test_inaccessible_driver_is_rejected(monkeypatch):
    fake_driver(monkeypatch, init=801)
    with pytest.raises(RuntimeError, match='CUDA error 801'):
        gpu.require_cuda()


def test_no_gpu_is_rejected(monkeypatch):
    fake_driver(monkeypatch, count=0)
    with pytest.raises(RuntimeError, match='devices=0'):
        gpu.require_cuda()


def test_accessible_gpu_is_accepted(monkeypatch):
    fake_driver(monkeypatch)
    assert gpu.require_cuda() == 1
