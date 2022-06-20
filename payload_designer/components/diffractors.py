"""Diffractor component classes."""

# stdlib
import logging

# external
import numpy as np

# project
from payload_designer.components import Component
from payload_designer.libs.physlib import snell
from payload_designer.luts import LUT

# import math

# import pandas as pd


# import math
# from re import M

# import pandas as pd


LOG = logging.getLogger(__name__)


class SRTGrating(Component):
    """Surface-relief transmissive diffraction grating component."""

    def __init__(
        self,
        fringe_frequency=None,
        transmittance: LUT = None,
        mass=None,
        dimensions=None,
    ):
        super().__init__(mass=mass, dimensions=dimensions)
        self.fringe_frequency = fringe_frequency
        self.transmittance = transmittance

    def get_emergent_angle(self, incident_angle, wavelength, order=1):
        """Get the angle of the diffracted ray at the specified wavelengh.

        Args:
            incident_angle: Angle of collimated light incident on grating.
            wavelength: Wavelength of incident light.
            order: Diffraction order.

        """

        emergent_angle = np.arcsin(
            np.sin(incident_angle) - order * self.fringe_frequency * wavelength
        )

        return emergent_angle

    def get_illuminated_fringe_count(self, beam_diameter):
        """Get the number of fringes that are illuminated by an incident
        collimated beam of light."""
        assert self.fringe_frequency is not None, "Fringe frequency must be specified."

        fringe_count = beam_diameter * self.fringe_frequency

        return fringe_count

    def get_resolvance(self, beam_diameter, order=1):
        """Get the resolving power of the grating."""

        resolvance = order * self.get_illuminated_fringe_count(beam_diameter)

        return resolvance

    def get_dispersion(self, wavelength, incident_angle, order=1):
        """Get the angular dispersion of the grating."""

        emergent_angle = self.get_emergent_angle(
            incident_angle=incident_angle, wavelength=wavelength, order=order
        )

        dispersion = (np.sin(incident_angle) + np.sin(emergent_angle)) / (
            wavelength * np.cos(emergent_angle)
        )

        return dispersion

    def get_resolution(self, wavelength, beam_diameter, order=1):
        """Get the resolution of the grating.

        This is the optically-limited spectral resolution of a system.

        """

        resolution = wavelength / self.get_resolvance(
            beam_diameter=beam_diameter, order=order
        )

        return resolution

    def get_anamorphic_amplification(self, incident_angle, wavelength, order=1):
        """Get the anamorphic amplification of the grating."""

        emergent_angle = self.get_emergent_angle(
            incident_angle=incident_angle, wavelength=wavelength, order=order
        )

        emergent_to_incident_beam_width = np.cos(emergent_angle) / np.cos(
            incident_angle
        )

        return emergent_to_incident_beam_width

    def get_emergent_beam_width(self, beam_width, incident_angle, wavelength, order=1):
        """Get the width of the diffracted beam at the specified wavelength."""

        emergent_beam_width = (
            self.get_anamorphic_amplification(
                incident_angle=incident_angle, wavelength=wavelength, order=order
            )
            * beam_width
        )

        return emergent_beam_width


class VPHGrating(SRTGrating):
    """Volume-phase holographic grating component. Model assumes fringes are
    unslanted relative to the grating normal.

    Args:
        transmittance: Transmittance of the grating.
        mass: Mass of the grating.
        index_dcg_amplitude: Semiamplitude of the refractive-index modulation in the DCG.

    """

    def __init__(
        self,
        transmittance: LUT = None,
        mass=None,
        dimensions=None,
        fringe_frequency=None,
        index_seal=None,
        index_dcg=None,
        dcg_thickness=None,
        index_dcg_amplitude=None,
    ):
        super().__init__(
            fringe_frequency=fringe_frequency,
            transmittance=transmittance,
            mass=mass,
            dimensions=dimensions,
        )
        self.index_seal = index_seal
        self.index_dcg = index_dcg
        self.dcg_thickness = dcg_thickness
        self.index_dcg_amplitude = index_dcg_amplitude

    def get_emergent_angle(
        self, incident_angle, wavelength, n_initial=1, n_final=1, order=1
    ):

        angle_1 = snell(angle=incident_angle, n_1=n_initial, n_2=self.index_seal)
        angle_2 = np.arscin(
            (order * wavelength * self.fringe_frequency / self.index_dcg)
            - np.sin(angle_1)
        )
        angle_3 = snell(angle=angle_2, n_1=self.index_dcg, n_2=n_final)

        return angle_3

    def get_transmittance_theoretical(self, wavelength, order=1):
        """Calculates the Kogelnik efficiency for unpolarized light."""

        a_2b = np.arcsin((order * wavelength) / (2 * self.index_dcg * wavelength))
        mu_s = (
            np.sin(
                (np.pi * self.index_dcg_amplitude * self.dcg_thickness)
                / (wavelength * np.cos(a_2b))
            )
            ** 2
            + 0.5
            * np.sin(
                (
                    np.pi
                    * self.index_dcg_amplitude
                    * self.dcg_thickness
                    * np.cos(2 * a_2b)
                )
                / (wavelength * np.cos(a_2b))
            )
            ** 2
        )

        return mu_s

    def get_efficiency_bandwidth(self, wavelength, order=1):
        """Calculates the efficiency bandwidth."""
        a_2b = np.arcsin((order * wavelength) / (2 * self.index_dcg * wavelength))
        lmda_eff = (wavelength * wavelength) / (self.dcg_thickness * np.tan(a_2b))

        return lmda_eff


class VPHGrism(VPHGrating):
    """Volume-phase holographic grism component.

    Assumes symmetrical grism model.

    """

    def __init__(
        self,
        transmittance: LUT = None,
        mass=None,
        dimensions=None,
        fringe_frequency=None,
        dcg_thickness=None,
        index_dcg_amplitude=None,
        apex_angle=None,
        index_prism=None,
        index_seal=None,
        index_dcg=None,
    ):
        super().__init__(
            transmittance=transmittance,
            mass=mass,
            dimensions=dimensions,
            fringe_frequency=fringe_frequency,
            dcg_thickness=dcg_thickness,
            index_dcg_amplitude=index_dcg_amplitude,
            index_seal=index_seal,
            index_dcg=index_dcg,
        )
        self.index_prism = index_prism
        self.apex_angle = apex_angle

    def get_emergent_angle(
        self, incident_angle, wavelength, n_initial=1, n_final=1, order=1
    ):
        angle_1 = incident_angle + self.apex_angle
        angle_2 = snell(angle=angle_1, n_1=n_initial, n_2=self.index_prism)
        angle_3 = self.apex_angle - angle_2
        angle_4 = snell(angle=angle_3, n_1=self.index_prism, n_2=self.index_seal)
        angle_5 = angle_4
        angle_6 = np.arcsin(
            np.sin(angle_5) - order * self.fringe_frequency * wavelength
        )
        angle_7 = angle_6
        angle_8 = snell(angle=angle_7, n_1=self.index_seal, n_2=self.index_prism)
        angle_9 = angle_8 - self.apex_angle
        angle_10 = snell(angle=angle_9, n_1=self.index_prism, n_2=n_final)
        angle_out = angle_10 + self.apex_angle

        return angle_out

    def get_undeviated_wavelength(self, angle_in, order=1, index_in=1):
        """Calculates the undeviated wavelength."""
        angle_1 = angle_in + self.apex_angle
        angle_2 = snell(angle=angle_1, n_1=index_in, n_2=self.index_prism)
        angle_3 = self.apex_angle - angle_2
        angle_4 = snell(angle=angle_3, n_1=self.index_prism, n_2=self.index_seal)
        angle_5 = angle_4
        l_g = 2 * (np.sin(angle_5) / (order * self.fringe_frequency))

        return l_g

    def get_transmittance_theoretical(self):
        return NotImplementedError
        # angle_1 = a_in + a
        # angle_2 = physlib.snell_angle_2(angle_1=angle_1, n_1=n_1, n_2=n_2)
        # angle_3 = a - angle_2
        # angle_4 = physlib.snell_angle_2(angle_1=angle_3, n_1=n_2, n_2=n_3)
        # angle_5 = angle_4
        # L = 1 / v  # nm/lines

        # # diffraction efficiency
        # n_p = (np.sin((math.pi * n_g * d) / (l * np.cos(angle_5))) ** 2) + (
        #     (1 / 2)
        #     * (
        #         np.sin(
        #             ((math.pi * n_g * d) * np.cos(2 * angle_5)) / (l * np.cos(angle_5))
        #         )
        #     )
        #     ** 2
        # )  # angle_5 being close to bragg angle = more efficiency
        # n_p = n_p * eff_mat * eff_mat

        # return n_p


def get_efficiency(
    fringe_density,
    n_grism,
    n_grism_semiamp,
    grism_thickness,
    grism_length,
    incident_angle,
    order=1,
    n_air=1,
):
    """Calculates the efficiency for the VPH Grism under the Bragg Condition.

    Parameters
    ----------
    fringe_density : float
        The number of fringes per unit length of the grism.
    n_grism : float
        The refractive index of the grism.
    n_grism_semiamp : float
        Semiamplitude (half of peak-to-peak amplitude) of variation
        of n_grism inside grating volume.
    grism_thickness : float
        The thickness of the grism.
    grism_length : float
        The length of the grism.
    incident_angle : float
        The angle of incidence of the light on the grism.
    order : int, optional
        The order of the grating. The default is 1.
    n_air : float, optional
        The refractive index of the air. The default is 1.

    Returns
    -------
    float
        The efficiency of the grism.

    """
    sin_arg_num = order * np.pi * n_grism_semiamp * grism_thickness
    fringe_spacing = grism_length**2 / fringe_density
    sin_arg_den = (
        2
        * fringe_spacing
        * n_grism
        * np.sin(2 * np.arcsin(n_air / n_grism * np.sin(incident_angle)))
    )
    efficiency = (np.sin(sin_arg_num / sin_arg_den)) ** 2
    return efficiency
