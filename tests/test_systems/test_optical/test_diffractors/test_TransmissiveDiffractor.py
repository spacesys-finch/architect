"""Tests for transmissive diffractor component."""

# stdlib
import logging

# external
import astropy.units as unit

# project
from architect.systems.optical.diffractors import TransmissiveDiffractor

LOG = logging.getLogger(__name__)


def test_get_diffraction_angle():
    """Test get_diffraction_angle."""
    diffractor = TransmissiveDiffractor(fringe_frequency=600 * 1 / unit.m)
    angle = diffractor.get_diffraction_angle(wavelength=900 * unit.nm)

    LOG.info(f"diffraction angle: {angle}")
