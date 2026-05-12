import numpy as np
import pytest

from seabreeze_live import MockDevice, TriggerMode


def _fast_mock(**kwargs):
    return MockDevice(simulate_exposure=False, **kwargs)


def test_default_shape_and_metadata():
    d = _fast_mock()
    assert d.pixels == 2048
    assert d.wavelengths().shape == (2048,)
    assert d.serial_number.startswith("MOCK")
    assert d.model == "MockSpectrometer"


def test_integration_time_bounds():
    d = _fast_mock()
    lo, hi = d.integration_time_limits_us
    d.set_integration_time(lo)
    assert d.integration_time_us == lo
    d.set_integration_time(hi)
    assert d.integration_time_us == hi
    with pytest.raises(ValueError):
        d.set_integration_time(lo - 1)
    with pytest.raises(ValueError):
        d.set_integration_time(hi + 1)


def test_intensities_shape_and_bounds():
    d = _fast_mock()
    spec = d.read_intensities()
    assert spec.shape == (d.pixels,)
    assert spec.dtype == np.float64
    assert spec.min() >= 0.0
    assert spec.max() <= d.max_intensity


def test_intensities_change_per_call():
    d = _fast_mock()
    a = d.read_intensities()
    b = d.read_intensities()
    assert not np.array_equal(a, b)  # noise differs


def test_trigger_mode_only_normal_supported():
    d = _fast_mock()
    d.set_trigger_mode(TriggerMode.NORMAL)
    with pytest.raises(NotImplementedError):
        d.set_trigger_mode(TriggerMode.EXTERNAL_HARDWARE_EDGE)


def test_seed_reproducibility():
    a = _fast_mock(seed=42).read_intensities()
    b = _fast_mock(seed=42).read_intensities()
    np.testing.assert_array_equal(a, b)
