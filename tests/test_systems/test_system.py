"""System class tests."""

# stdlib
import logging

# project
from architect.systems import System

LOG = logging.getLogger(__name__)


def test_init():
    """Test initialization."""
    system = System()
    LOG.info(system)


def test_get_attrs_table():
    """Test get_attrs_table method."""
    system = System()

    system.some_property = "some_value"

    table = system.get_attrs_table()

    LOG.info(f"Attribute table:\n{table}")


def test_to_latex():
    """Test to_latex method."""
    system = System()

    system.some_property = "some_value"

    table = system.to_latex()

    LOG.info(f"Attribute LaTeX table:\n{table}")
