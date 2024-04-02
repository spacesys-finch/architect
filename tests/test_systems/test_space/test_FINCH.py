"""Satellite class tests."""

# stdlib
import logging

# project
from architect.systems.space.satellites import FINCH

LOG = logging.getLogger(__name__)


def test_init():
    """Test initialization."""
    satellite = FINCH()
    LOG.info(satellite)


def test_get_dimensions():
    """Test get_dimensions() method."""
    satellite = FINCH()

    result = satellite.get_dimensions()
    LOG.info(result)
