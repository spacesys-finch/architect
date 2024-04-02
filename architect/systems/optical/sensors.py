"""Sensor component classes."""

# stdlib
import logging
import math

# external
import astropy.units as unit
import numpy as np
from astropy.units import Quantity

# project
from architect import luts
from architect.luts import LUT
from architect.systems import Component

LOG = logging.getLogger(__name__)


class Sensor(Component):
    """Generic focal plane array sensor component.

    Args:
        dimensions: Dimensions of component bounding box. Elements are ordered as (x, y, z) in the cubesat frame.
        integration_time: Integration time.
        efficiency: Quantum efficiency of the sensor.
        i_dark: Dark current.
        mass: Component mass.
        n_bin: Number of binning operations performed on image aquisition.
        n_bit: Sensor bit depth.
        n_px: Pixel count in the (x, y) dimensions in cubesat frame.
        n_well: Sensor well depth.
        noise_read: Readout noise.
        pitch: Pixel pitch. The distance between the centerpoints of adjacent pixels.
        waveband: Waveband. The range of wavelengths the sensor is able to measure.

    """

    def __init__(
        self,
        dimensions: tuple[Quantity[unit.m], Quantity[unit.m], Quantity[unit.m]] = None,
        integration_time: Quantity[unit.s] = None,
        efficiency: LUT = None,
        i_dark: unit.electron / unit.pix / unit.s = None,
        mass: unit.kg = None,
        n_bin=None,
        n_bit=None,
        n_px: tuple[int, int] = None,
        n_well=None,
        noise_read=None,
        pitch: unit.m = None,
        waveband: unit.m = None,
    ):
        super().__init__(dimensions=dimensions, mass=mass)
        self.integration_time = integration_time
        self.efficiency = efficiency
        self.i_dark = i_dark
        self.n_bin = n_bin
        self.n_bit = n_bit
        self.n_px = n_px
        self.n_well = n_well
        self.noise_read = noise_read
        self.waveband = waveband
        self.pitch = pitch

    def get_pitch(self):
        """Get the pixel pitch."""
        if self.pitch is not None:
            return self.pitch
        else:
            raise ValueError("Pixel pitch not set.")

    def get_n_px(self):
        """Get the pixel count."""
        if self.n_px is not None:
            return self.n_px
        else:
            raise ValueError("Pixel count not set.")

    def get_n_bin(self):
        """Get the number of binning operations."""
        if self.n_bin is not None:
            return self.n_bin
        else:
            raise ValueError("bin is not set.")

    def get_shape(self) -> tuple:
        """Get the dimensions of the sensor face."""
        assert self.n_px is not None, "n_px must be specified."
        assert self.pitch is not None, "Pitch must be specified."

        size = (self.n_px[0] * self.pitch, self.n_px[1] * self.pitch)

        return size

    def get_noise_read(self):
        """Get the noise read."""
        if self.noise_read is not None:
            return self.noise_read
        else:
            raise ValueError("Noise read not set.")

    def get_area(self):
        """Get the area of the sensor face."""

        shape = self.get_shape()

        area = shape[0] * shape[1]

        return area

    def get_pixel_area(self):
        """Get the area of a single detector element (pixel).

        Assumes square pixels.

        """
        assert self.pitch is not None, "Pitch must be specified."

        pixel_area = self.pitch**2

        return pixel_area

    def get_mean_dark_signal(self) -> Quantity[unit.electron / unit.pix]:
        """Get the mean dark signal.

        Ref: https://www.notion.so/utat-ss/Mean-Dark-Signal-55519f6c43654fae9464b578da2965d9

        """
        assert self.i_dark is not None, "i_dark must be specified."
        assert self.integration_time is not None, "Integration time must be specified."

        dark_noise = self.i_dark * self.integration_time

        return dark_noise

    def get_dark_shot_noise(self):
        """Get the dark shot noise.

        Ref: https://www.notion.so/utat-ss/Dark-Shot-Noise-d0632bd2a0444d7eb814beed1224ba06

        """

        dark_shot_noise = np.sqrt(self.get_mean_dark_signal())

        return dark_shot_noise

    def get_quantization_noise(self):
        """Get the quantization noise of the sensor."""
        assert self.n_well is not None, "n_well must be specified."
        assert self.n_bit is not None, "n_bit must be specified."

        quant_noise = (1 / math.sqrt(12)) * self.n_well / 2**self.n_bit.value

        return quant_noise

    def get_noise(self, signal):
        """Get the net noise of the sensor."""
        assert self.n_bin is not None, "n_bin must be specified."
        assert self.noise_read is not None, "noise_read must be specified."

        noise = np.sqrt(
            signal * unit.electron
            + self.n_bin * (self.get_mean_dark_signal() * unit.pix) ** 2
            + self.get_quantization_noise() ** 2
            + self.n_bin * self.noise_read**2
        )

        return noise

    def get_integration_time(self) -> Quantity[unit.s]:
        """Get the integration time."""
        if self.integration_time is not None:
            return self.integration_time
        else:
            raise ValueError("Integration time is not set.")

    def get_efficiency(self, wavelength: Quantity[unit.m]):
        """Get the quantum efficiency of the sensor."""
        if self.efficiency is not None:
            return self.efficiency(wavelength)
        else:
            raise ValueError("Quantum efficiency is not set.")

    def get_waveband(self) -> Quantity[unit.m]:
        """Get the waveband of the sensor."""
        if self.waveband is not None:
            return self.waveband
        else:
            raise ValueError("Waveband is not set.")

    def get_datacube_size(self) -> unit.bit:
        """Get the size of datacube in bit.

        Ref: https://www.notion.so/utat-ss/Hyperspectral-Datacube-Size-5390b2593dff48b2b96d954eb8267844

        """

        assert self.n_px is not None, "n_px must be specified."
        assert self.n_bit is not None, "n_bit must be specified."

        npxx = self.n_px[
            0
        ]  # 640 px, pixels on the sensor in the across-track direction -> spatial information
        npxy = self.n_px[
            1
        ]  # 512 px, pixels on the sensor in the along-track direction -> spectral information
        nbit = self.n_bit  # bit depth
        D = npxx * npxx * npxy * nbit
        return D / unit.pix**3


class TauSWIR(Sensor):
    """Teledyne FLIR Tau SWIR sensor.

    Ref: https://www.notion.so/utat-ss/FLIR-Tau-SWIR-407ea145ffea4d188c777f3b17a9be2b

    """

    def __init__(self):
        ke = 1e3 * unit.electron
        super().__init__(
            dimensions=(38, 38, 36) * unit.mm,
            integration_time=166.7 * unit.ms,
            efficiency=luts.load("sensors/tauswir_quantum_efficiency"),
            i_dark=140 * (ke / unit.pix / unit.s),
            mass=81 * unit.g,
            n_bin=1 * unit.dimensionless_unscaled,
            n_bit=14 * unit.bit,
            n_px=(640, 512) * unit.pix,
            n_well=19 * ke,
            noise_read=500 * unit.electron,
            pitch=15 * unit.um,
            waveband=800 * unit.nm,
        )
