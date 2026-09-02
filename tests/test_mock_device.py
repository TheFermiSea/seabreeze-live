import numpy as np
import pytest

import seabreeze_live.device as device_module
from seabreeze_live import (
    HardwareConnectionError,
    HardwareOperationError,
    MockDevice,
    TriggerMode,
)


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


class _FakeSpectrometer:
    serial_number = "USB2+F01234"
    model = "USB2000PLUS"
    integration_time_micros_limits = (1_000, 20_000)

    def __init__(self):
        self.integration_time = None
        self.last_intensity_kwargs = None
        self.closed = False

    @classmethod
    def from_serial_number(cls, serial):
        assert serial == cls.serial_number
        return cls()

    @classmethod
    def from_first_available(cls):
        return cls()

    def wavelengths(self):
        return np.array([400.0, 500.0, 600.0])

    def integration_time_micros(self, value):
        self.integration_time = value

    def intensities(self, **kwargs):
        self.last_intensity_kwargs = kwargs
        return np.array([1.0, 2.0, 3.0])

    def close(self):
        self.closed = True


def test_real_hardware_path_selects_serial_and_uses_seabreeze_keywords(monkeypatch):
    backend = type("Backend", (), {"Spectrometer": _FakeSpectrometer})
    monkeypatch.setattr(device_module, "SEABREEZE_AVAILABLE", True)
    monkeypatch.setattr(device_module, "sb", backend)

    with device_module.SpectrometerDevice(_FakeSpectrometer.serial_number) as device:
        assert device.meta is not None and not device.meta.is_mock
        assert device.integration_time_us == 20_000  # default clamped to limits
        np.testing.assert_array_equal(device.get_wavelengths(), [400.0, 500.0, 600.0])
        np.testing.assert_array_equal(
            device.get_intensities(correct_dark_pixels=True), [1, 2, 3]
        )
        assert device.device.last_intensity_kwargs == {
            "correct_dark_counts": True,
            "correct_nonlinearity": False,
        }
        raw = device.device
    assert raw.closed is True


def test_real_hardware_never_silently_falls_back_to_mock(monkeypatch):
    monkeypatch.setattr(device_module, "SEABREEZE_AVAILABLE", False)
    with pytest.raises(
        HardwareConnectionError, match="python-seabreeze is unavailable"
    ):
        device_module.SpectrometerDevice()


def test_hardware_control_errors_are_not_silently_suppressed(monkeypatch):
    class FailingTrigger(_FakeSpectrometer):
        def trigger_mode(self, _mode):
            raise OSError("USB write failed")

    backend = type("Backend", (), {"Spectrometer": FailingTrigger})
    monkeypatch.setattr(device_module, "SEABREEZE_AVAILABLE", True)
    monkeypatch.setattr(device_module, "sb", backend)

    with (
        device_module.SpectrometerDevice() as device,
        pytest.raises(HardwareOperationError, match="trigger mode"),
    ):
        device.set_trigger_mode(TriggerMode.NORMAL)
