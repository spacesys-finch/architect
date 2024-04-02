"""Tests for VPH Grating component."""

# stdlib
import logging

# external
import astropy.units as unit

# project
from architect.systems.optical.diffractors import VPHGrating

LOG = logging.getLogger(__name__)


def test_init():
    """Test init method."""

    grating = VPHGrating()
    LOG.info(grating)


def test_get_diffraction_angle():
    """Test get_diffraction_angle."""

    grating = VPHGrating(
        index_dcg=20, fringe_frequency=600 * (1 / unit.mm), index_seal=1.0
    )
    result = grating.get_diffraction_angle(
        incident_angle=0 * unit.rad, wavelength=1300 * unit.nm
    )
    LOG.info(result)

    assert result.unit == unit.rad
