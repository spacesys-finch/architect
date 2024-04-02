"""Tests for VPH Grism component."""

# stdlib
import logging

# external
import astropy.units as unit

# project
from architect.systems.optical.diffractors import VPHGrism

LOG = logging.getLogger(__name__)


def test_init():
    """Test init method."""

    grism = VPHGrism()
    LOG.info(grism)


def test_get_diffraction_angle():
    """Test get_diffraction_angle method."""

    grism = VPHGrism(
        apex_angle=15 * unit.deg,
        index_prism=1.0,
        fringe_frequency=600 * (1 / unit.mm),
        index_seal=1.0,
    )

    result = grism.get_diffraction_angle(
        incident_angle=0 * unit.deg, wavelength=1600 * unit.nm
    )
    LOG.info(result)

    assert result.unit == unit.rad
