"""Satellite class tests."""

# stdlib
import logging

# external
import astropy.units as unit

# project
from architect.systems.space.satellites import CubeSat

LOG = logging.getLogger(__name__)


def test_init():
    """Test initialization."""
    satellite = CubeSat()
    LOG.info(satellite)


def test_get_volume():
    """Test get_dimensions() method."""
    satellite = CubeSat(units=3)

    result = satellite.get_volume()
    LOG.info(result)

    assert result.decompose().unit == unit.m**3
