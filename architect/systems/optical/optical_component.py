"""Optical component class."""

# stdlib
import logging

# external
import astropy.units as unit
from astropy.units import Quantity

# project
from architect.luts import LUT
from architect.systems import Component, System

LOG = logging.getLogger(__name__)


class OpticalComponent(Component):
    """A component with optical properties."""

    def __init__(
        self,
        dimensions: tuple[unit.m, unit.m, unit.m] = None,
        mass: unit.kg = None,
        volume: unit.m**3 = None,
        density: unit.kg / unit.m**3 = None,
        index=None,
        transmittance: Quantity[unit.pct] | LUT = None,
        **systems: System
    ):
        super().__init__(
            dimensions=dimensions, mass=mass, volume=volume, density=density, **systems
        )
        self.index = index
        self.transmittance = transmittance

    def get_transmittance(self, wavelength: unit.m = None) -> unit.pct:
        """Get the transmittance.

        Assumes a linear optical path.

        """
        if self.transmittance is not None:
            if isinstance(self.transmittance, LUT):
                assert wavelength is not None, "wavelength must be set."
                transmittance = self.transmittance(wavelength)
            else:
                transmittance = self.transmittance
        else:
            transmittance = 100 * unit.pct
            for system in self.systems:
                if isinstance(system, OpticalComponent):
                    transmittance = transmittance * system.get_transmittance(wavelength)

        return transmittance.to(unit.pct)

    def get_index(self):
        """Get the index of refraction."""
        if self.index is not None:
            return self.index
        else:
            raise ValueError("index is not set.")
