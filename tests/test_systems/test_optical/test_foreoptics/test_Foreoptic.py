"""Tests for Foreoptic component."""

# stdlib
import logging

# external
import astropy.units as unit

# project
from architect.systems.optical.foreoptics import Foreoptic

LOG = logging.getLogger(__name__)


def test_init():
    """Test init method."""

    foreoptic = Foreoptic()
    LOG.info(foreoptic)


def test_get_image_area():
    """Test get_image_area method."""

    foreoptic = Foreoptic(image_diameter=20 * unit.mm)

    result = foreoptic.get_image_area()

    LOG.info(result)

    assert result.decompose().unit == unit.m**2


def test_get_f_number():
    """Test get_f_number method."""

    foreoptic = Foreoptic(focal_length=100 * unit.mm, diameter=80 * unit.mm)

    result = foreoptic.get_f_number()
    LOG.info(result)

    assert result.unit == unit.dimensionless_unscaled
