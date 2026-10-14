import numpy as np

from cloelib.auxiliary.halo_helpers import convert_to_Delta_crit
from cloelib.observables.halos.halo_model_properties import HaloModelProperties

from .halo_abundance_base import HaloAbundanceBase


class TinkerHaloAbundance(HaloAbundanceBase):
    """
    Class implementing the Tinker et al. mass abundance models.

    Following the cold dark matter
    prescription by Costanzi+13 (https://arxiv.org/abs/1311.1514) and
    Castorina+13 (https://arxiv.org/pdf/1311.1212), the halo mass function
    and halo bias do not include the massive neutrino contribution in the
    computation of mass variance, power spectrum, and overdensity.
    """

    def __init__(
        self,
        halo_model_properties: HaloModelProperties,
        overdensity_type: str = "vir",
        overdensity: int = 200,
    ):
        HaloAbundanceBase.__init__(self, halo_model_properties)
        self.overdensity_type = overdensity_type
        self.overdensity = overdensity

    def f_sigma_nu(self, z, M):
        r"""
        Computation of the multiplicity function.

        Computes the Tinker et al. (2008) multiplicity function
        at the requested redshift and mass points. The parameters are
        interpolated in the overdensity with respect to the mean density,
        calibrated for 200 <= Delta_m <= 3200 and z <~ 2.5. A ValueError is
        raised for Delta_m outside this range.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        f_sigma_nu: numpy.ndarray
            f_sigma_nu[i,j], where i is the redshift axis and j the mass axis
        """
        # compute inputs
        sigma = self.sigma_z_M(z, M)
        Delta = convert_to_Delta_crit(
            self.overdensity_type,
            self.overdensity,
            self.halo_model_properties.background,
            z,
        ) / self.halo_model_properties.background.Omega_cb(z)
        Delta = np.atleast_1d(Delta)

        ##################
        # HMF computations
        ##################

        # parameters (Table 2 of Tinker et al. 2008), as a function of the
        # overdensity with respect to the mean density
        Delta_tab = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
        A0_tab = np.array(
            [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260]
        )
        a0_tab = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        b0_tab = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        c0_tab = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])

        if np.any(Delta < Delta_tab[0]) or np.any(Delta > Delta_tab[-1]):
            raise ValueError(
                "Tinker et al. (2008) is calibrated for 200 <= Delta_m <= 3200; "
                f"got Delta_m in [{Delta.min():.1f}, {Delta.max():.1f}]."
            )

        # Compute main quantities
        A0 = np.interp(Delta, Delta_tab, A0_tab)
        a0 = np.interp(Delta, Delta_tab, a0_tab)
        b0 = np.interp(Delta, Delta_tab, b0_tab)
        c = np.interp(Delta, Delta_tab, c0_tab)
        alpha = 10.0 ** (-((0.75 / np.log10(Delta / 75.0)) ** 1.2))

        A = A0 * (1.0 + np.asarray(z)) ** -0.14
        a = a0 * (1.0 + np.asarray(z)) ** -0.06
        b = b0 * (1.0 + np.asarray(z)) ** -alpha

        return (
            A[:, np.newaxis]
            * ((sigma / b[:, np.newaxis]) ** -a[:, np.newaxis] + 1.0)
            * np.exp(-c[:, np.newaxis] / sigma**2.0)
        )

    def bias(self, z, M):
        r"""
        Computation of the halo bias.

        Computes the Tinker et al. (2010) halo bias
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points
        M: numpy.ndarray
            Mass points in h^{-1} Msun

        Returns
        -------
        bias: numpy.ndarray
            bias[i,j], where i is the redshift axis and j the mass axis
        """
        # compute inputs
        delta_c = self.delta_c(z)
        nu = self.nu_z_M(z, M)
        Delta = convert_to_Delta_crit(
            self.overdensity_type,
            self.overdensity,
            self.halo_model_properties.background,
            z,
        ) / self.halo_model_properties.background.Omega_cb(z)

        ###################
        # Bias computations
        ###################

        # parameters
        p = [1.0, 0.24, 0.44, 0.88, 0.183, 1.5, 0.019, 0.107, 0.19, 2.4]
        y = np.log10(Delta)
        A_par = p[0] + p[1] * y * np.e ** (-((4.0 / y) ** 4))
        a_par = p[2] * y - p[3]
        B_par = p[4]
        b_par = p[5]
        C_par = p[6] + p[7] * y + p[8] * np.e ** (-((4.0 / y) ** 4))
        c_par = p[9]

        # bias
        _nu = nu.T
        return (
            1.0
            - A_par * _nu**a_par / (_nu**a_par + delta_c**a_par)
            + B_par * _nu**b_par
            + C_par * _nu**c_par
        ).T
