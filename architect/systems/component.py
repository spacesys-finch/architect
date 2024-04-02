"""Component class."""

# external
import astropy.units as unit

# project
from architect.systems import System


class Component(System):
    """A component is a physical system in three-dimensional space that may
    contain nested systems."""

    def __init__(
        self,
        dimensions: tuple[unit.m, unit.m, unit.m] = None,
        mass: unit.kg = None,
        volume: unit.m**3 = None,
        density: unit.kg / unit.m**3 = None,
        **systems: System
    ):
        super().__init__(**systems)
        self.dimensions = dimensions
        self.mass = mass
        self.volume = volume
        self.density = density

    def get_dimensions(self):
        """Get the dimensions of the component."""
        if self.dimensions is not None:
            dimensions = self.dimensions
        else:
            raise ValueError("Dimensions must be specified.")

        return dimensions

    def get_volume(self):
        """Get the volume of the component."""
        if self.volume is not None:
            volume = self.volume
        else:
            volume = (
                self.get_dimensions()[0]
                * self.get_dimensions()[1]
                * self.get_dimensions()[2]
            )

        return volume

    def get_mass(self):
        """Get the mass of the component."""
        if self.mass is not None:
            mass = self.mass
        else:
            mass = 0
            for system in self.systems:
                mass += system.get_mass()

        return mass

    def get_density(self):
        """Get the density of the component.

        Ref: https://www.notion.so/utat-ss/Density-5bb20fc84e2644feb7c668e2c620c962

        """
        if self.density is not None:
            density = self.density
        else:
            density = self.get_mass() / self.get_volume()

        return density
