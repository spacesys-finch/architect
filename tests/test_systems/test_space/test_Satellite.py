"""Satellite class tests."""

# stdlib
import logging

# external
import astropy.units as unit

# project
from architect.systems.space.satellites import Satellite

LOG = logging.getLogger(__name__)


def test_init():
    """Test initialization."""
    satellite = Satellite()
    LOG.info(satellite)


def test_get_orbit_radius():
    """Test get_orbit_radius method."""

    satellite = Satellite(altitude=500 * unit.km)

    result = satellite.get_orbit_radius()
    LOG.info(result)

    assert result.unit == unit.m


def test_get_orbit_velocity():
    """Test get_orbit_velocity method."""

    satellite = Satellite(altitude=500 * unit.km)
    result = satellite.get_orbit_velocity()
    LOG.info(result)

    assert result.unit == unit.m / unit.s


def test_get_orbit_angular_velocity():
    """Test get_orbit_angular_velocity method."""

    satellite = Satellite(altitude=500 * unit.km)

    result = satellite.get_orbit_angular_velocity()
    LOG.info(result)

    assert result.unit == unit.rad / unit.s


def test_get_orbit_ground_projected_velocity():
    """Test get_orbit_ground_projected_velocity method."""

    satellite = Satellite(altitude=500 * unit.km)

    result = satellite.get_orbit_ground_projected_velocity()
    LOG.info(result)

    assert result.unit == unit.m / unit.s
