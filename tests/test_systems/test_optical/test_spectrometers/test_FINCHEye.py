"""FINCHEye class tests."""

# stdlib
import logging

# external
import astropy.units as unit

# project
from architect.systems.optical.diffractors import TransmissiveDiffractor
from architect.systems.optical.foreoptics import Foreoptic
from architect.systems.optical.lenses import Lens
from architect.systems.optical.spectrometers import FINCHEye

LOG = logging.getLogger(__name__)


def test_init():
    """Test init method."""

    payload = FINCHEye()
    LOG.info(payload)


def test_get_dimensions():
    """Test get_dimensions method."""

    payload = FINCHEye(foreoptic=Foreoptic(diameter=50 * unit.mm, length=60 * unit.mm))

    result = payload.get_dimensions()
    LOG.info(result)

    assert result[0].decompose() == 50 * unit.mm.decompose()
    assert result[1].decompose() == 50 * unit.mm.decompose()
    assert result[2].decompose() == 96 * unit.mm.decompose()


def test_get_sensor_wavelength_mapping():
    """Test get_sensor_wavelength_mapping method."""

    payload = FINCHEye(
        grism=TransmissiveDiffractor(fringe_frequency=600 * (1 / unit.mm)),
        focuser=Lens(focal_length=10 * unit.mm),
    )

    result = payload.get_sensor_wavelength_mapping(wavelength=900 * unit.nm)
    LOG.info(result)

    assert result.decompose().unit == unit.m
