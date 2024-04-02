"""Satellite system classes."""

# external
import astropy.constants as const
import astropy.units as unit
import numpy as np
from astropy.units import Quantity

# project
from architect.systems import Component
from architect.systems.optical.spectrometers import FINCHEye


class Satellite(Component):
    """Base class for satellite systems."""

    def __init__(self, altitude: Quantity[unit.m] = None):
        super().__init__()
        self.altitude = altitude

    def get_orbit_radius(self):
        """Get the orbital radius.

        Ref: https://www.notion.so/utat-ss/Orbital-Radius-b07adc4a1a7543b2bceebf4fbeb61098

        """
        assert self.altitude is not None, "Altitude must be specified."

        R_orbit = const.R_earth + self.altitude

        return R_orbit

    def get_orbit_velocity(self):
        """Get the orbital velocity.

        Ref: https://www.notion.so/utat-ss/Orbital-Velocity-1cf0834326664872a1682db4bcd3a610

        """
        v_orbit = np.sqrt(const.G * const.M_earth / self.get_orbit_radius())

        return v_orbit

    def get_orbit_angular_velocity(self):
        """Get the orbital angular velocity.

        Ref: https://www.notion.so/utat-ss/Orbital-Angular-Velocity-40aba4f9348b4c01a0ae0ecd1ac17d8f

        """
        w_orbit = self.get_orbit_velocity() / self.get_orbit_radius() * unit.rad

        return w_orbit

    def get_orbit_ground_projected_velocity(self):
        """Get the orbital ground projected velocity.

        Ref: https://www.notion.so/utat-ss/Ground-Projected-Orbital-Velocity-4248ebec57634a42beebf619b0db1793

        """
        v_ground = (self.get_orbit_angular_velocity() / unit.rad) * const.R_earth

        return v_ground


class CubeSat(Satellite):
    """Miniaturized satellite based on a form factor consisting of 10cm
    cubes."""

    def __init__(self, altitude: Quantity[unit.m] = None, units: int = None):
        super().__init__(altitude=altitude)
        self.units = units

    def get_volume(self):
        """Get the volume of the CubeSat from its units (U's)."""
        assert self.units is not None, "CubeSat U's must be specified."

        volume = (100 * unit.mm) ** 3 * self.units

        return volume


class FINCH(CubeSat):
    """FINCH satellite system."""

    def __init__(self, payload: FINCHEye = None, altitude: Quantity[unit.m] = None):
        super().__init__(altitude=altitude, units=3)
        self.payload = payload

    def get_dimensions(self):
        """Get the bounding box dimensions of the CubeSat."""
        return (100 * unit.mm, 100 * unit.mm, self.units * (100 * unit.mm))
