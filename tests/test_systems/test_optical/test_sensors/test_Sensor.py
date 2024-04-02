"""Tests for Sensor component."""

# stdlib
import logging

# external
import astropy.units as unit

# project
from architect import luts
from architect.systems.optical.sensors import Sensor

LOG = logging.getLogger(__name__)


def test_init():
    """Test init method."""

    sensor = Sensor()
    LOG.info(sensor)


def test_get_pitch():
    """Test get_pitch method."""
    pitch = 10 * unit.um
    sensor = Sensor(pitch=pitch)

    result = sensor.get_pitch()
    LOG.info(result)

    assert result == pitch


def test_get_n_px():
    """Test get_n_px method."""

    n_px = (640, 512) * unit.pix
    sensor = Sensor(n_px=n_px)

    result = sensor.get_n_px()
    # LOG.info(result)

    assert (result == n_px).all()


def test_get_n_bin():
    """Test get_n_bin method."""

    n_bin = 4 * unit.dimensionless_unscaled
    sensor = Sensor(n_bin=n_bin)

    result = sensor.get_n_bin()
    LOG.info(result)

    assert result == n_bin


def test_get_shape():
    """Test get_shape method."""

    sensor = Sensor(n_px=(640, 512), pitch=10 * unit.um)

    result = sensor.get_shape()
    LOG.info(result)

    assert result[0].decompose().unit == unit.m
    assert result[1].decompose().unit == unit.m


def test_get_area():
    """Test get_area method."""

    sensor = Sensor(n_px=(640, 512), pitch=10 * unit.um)

    result = sensor.get_area()
    LOG.info(result)

    assert result.decompose().unit == unit.m**2


def test_get_pixel_area():
    """Test get_pixel_area method."""

    sensor = Sensor(pitch=10 * unit.um)

    result = sensor.get_pixel_area()
    LOG.info(result)

    assert result.decompose().unit == unit.m**2


def test_get_mean_dark_signal():
    """Test get_mean_dark_signal method."""

    ke = 1e3 * unit.electron
    sensor = Sensor(
        integration_time=166.7 * unit.ms, i_dark=140 * (ke / unit.pix / unit.s)
    )

    result = sensor.get_mean_dark_signal()
    LOG.info(result)

    assert result.decompose().unit == (unit.electron / unit.pix)


def test_get_quantization_noise():
    """Test get_quantization_noise method."""

    ke = 1e3 * unit.electron
    sensor = Sensor(n_well=19 * ke, n_bit=14 * unit.bit)

    result = sensor.get_quantization_noise()
    LOG.info(result)

    assert result.decompose().unit == unit.electron


def test_get_noise():
    """Test get_noise method."""

    ke = 1e3 * unit.electron
    signal = 6 * 10**6 * unit.electron
    sensor = Sensor(
        integration_time=166.7 * unit.ms,
        i_dark=140 * (ke / unit.pix / unit.s),
        n_well=19 * ke,
        n_bit=14 * unit.bit,
        n_bin=1 * unit.dimensionless_unscaled,
        noise_read=500 * unit.electron,
    )

    result = sensor.get_noise(signal)
    LOG.info(result)

    assert result.decompose().unit == unit.electron


def test_get_integration_time():
    """Test get_integration_time method."""

    integration_time = 166.7 * unit.ms
    sensor = Sensor(integration_time=integration_time)

    result = sensor.get_integration_time()
    LOG.info(result)

    assert result == integration_time


def test_get_efficiency():
    """Test get_efficiency method."""

    wavelength = 1200 * unit.nm
    sensor = Sensor(efficiency=luts.load("sensors/tauswir_quantum_efficiency"))

    result = sensor.get_efficiency(wavelength)
    LOG.info(result)

    assert result.unit == unit.pct * unit.electron


def test_get_datacube_size():
    """Test get_datacube_size method."""

    sensor = Sensor(n_px=(640, 512) * unit.pix, n_bit=14 * unit.bit)

    result = sensor.get_datacube_size()
    LOG.info(result)

    assert result.unit == unit.bit
