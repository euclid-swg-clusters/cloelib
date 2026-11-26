import numpy as np
from scipy.integrate import simpson
from scipy.stats import skewnorm

from cloelib.auxiliary import units
from cloelib.auxiliary.halo_helpers import (
    convert_distance,
    convert_to_Delta_crit,
)
from cloelib.cosmology import derived_cosmology
from cloelib.observables.halos.halo_model_properties import HaloModelProperties


class HaloProfileBase:
    def __init__(
        self,
        halo_model_properties: HaloModelProperties,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        zs_max: float = 4.0,
        mean_nz: np.ndarray | None = None,
        sigma_nz: np.ndarray | None = None,
        alpha_nz: np.ndarray | None = None,
        z_min_tomo_bin: np.ndarray | None = None,
    ):
        r"""Parent class computing quantities used in mass profile models.

        Initialize the class with given perturbations and overdensity definition.

        Parameters
        ----------
        halo_model_properties : HaloModelProperties
            An object from the `HaloModelProperties` class.
        mean_nz : np.ndarray
            Location parameter of the skew-normal redshift distribution of the
            sources in the tomographic bins >= t,
            shape (Number of tomographic source redshift bins,).
        sigma_nz : np.ndarray
            Scale parameter of the skew-normal redshift distribution of the
            sources in the tomographic bins >= t,
            shape (Number of tomographic source redshift bins,).
        alpha_nz : np.ndarray
            Skewness parameter of the skew-normal redshift distribution of the
            sources in the tomographic bins >= t,
            shape (Number of tomographic source redshift bins,).
        z_min_tomo_bin : np.ndarray
            Lower edge of each tomographic bin t, in increasing order,
            shape (Number of tomographic source redshift bins,), used to select
            the first tomographic bin of the source sample (see `get_tomo_bin_index`).

        Notes
        -----
        For a cluster with observed redshift z_obs, the source sample consists
        of the galaxies in the tomographic bins t, t+1, ..., where t is the first
        tomographic bin with lower edge larger than z_obs + Delta_z. In the model,
        z_obs is approximated by the true redshift of the cluster (see
        `sigma_crit_inv_eff`). Its redshift distribution is modeled with a
        skew-normal distribution of parameters (alpha_nz[t], mean_nz[t], sigma_nz[t]),
        normalized between 0 and zs_max: the sources in front of the cluster are
        part of the sample, with zero lensing signal.
        """
        self.halo_model_properties = halo_model_properties
        self.overdensity_type = overdensity_type
        self.overdensity = overdensity
        self.zs_max = zs_max
        self.mean_nz = np.atleast_1d(
            mean_nz
            if mean_nz is not None
            else np.array(
                [0.18739829, 0.33775482, 0.50338379, 0.66827123, 0.86322198, 1.30780699]
            )
        )
        self.sigma_nz = np.atleast_1d(
            sigma_nz
            if sigma_nz is not None
            else np.array(
                [0.85102341, 0.77623404, 0.74479123, 0.79814509, 0.85482975, 0.77782535]
            )
        )
        self.alpha_nz = np.atleast_1d(
            alpha_nz
            if alpha_nz is not None
            else np.array(
                [7.4511585, 3.12760121, 3.24293517, 6.36498593, 4.32074969, 1.50975262]
            )
        )
        self.z_min_tomo_bin = np.atleast_1d(
            z_min_tomo_bin
            if z_min_tomo_bin is not None
            else np.array([0.2398, 0.3117, 0.4304, 0.6960, 0.8876, 1.1267])
        )
        sizes = [
            par.size
            for par in (
                self.mean_nz,
                self.sigma_nz,
                self.alpha_nz,
                self.z_min_tomo_bin,
            )
        ]
        if len(set(sizes)) != 1:
            raise ValueError(
                "mean_nz, sigma_nz, alpha_nz and z_min_tomo_bin must have the same "
                f"size (number of tomographic bins), got {sizes}."
            )

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.halo_model_properties.background

    def sigma_crit(self, z, z_sources):
        r"""
        Critical surface mass density.

        Computes the critical surface mass density at the given
        lens and source redshifts.

        Parameters
        ----------
        z: float
            Lens redshift.
        z_sources: np.ndarray
            Source redshift points.

        Returns
        -------
        sigma_crit : float
            Critical surface mass density (unit: h * Msun / pc^2)
        """
        fact = (units.SPEED_OF_LIGHT / 1.0e3 / units.MPC_TO_KM) ** 2.0 / (
            4.0 * np.pi * units.GRAVITATIONAL_CONSTANT
        )  # Msun/Mpc
        d_a_sources = self.halo_model_properties.angular_diameter_distance(
            z_sources
        )  # Mpc
        d_m_sources = (1.0 + z_sources) * d_a_sources
        d_a_lens = self.halo_model_properties.angular_diameter_distance(z)[
            :, np.newaxis
        ]  # Mpc
        d_m_lens = (1.0 + z[:, np.newaxis]) * d_a_lens
        d_h = units.SPEED_OF_LIGHT / 1e3 / self.background.H0  # Mpc
        d_a_lens_source = (
            1.0
            / (1.0 + z_sources)
            * (
                d_m_sources
                * np.sqrt(1.0 + self.background.Omega_k0 * (d_m_lens / d_h) ** 2.0)
                - d_m_lens
                * np.sqrt(1.0 + self.background.Omega_k0 * (d_m_sources / d_h) ** 2.0)
            )
        )
        sig_crit = fact * d_a_sources / (d_a_lens * d_a_lens_source)

        return 1e-12 * sig_crit / self.background.h  # Msun pc^{-2} h

    def get_tomo_bin_index(self, z_obs, Delta_z=0.05):
        r"""
        Select the first tomographic bin of the source sample.

        Returns the first index t such that z_min_tomo_bin[t] > z_obs + Delta_z,
        i.e. the sources are the galaxies in the tomographic bins t, t+1, ...

        Parameters
        ----------
        z_obs : float
            Observed redshift of the cluster.
        Delta_z : float, optional
            Minimum redshift separation between the clusters and the lower edge of
            the tomographic bins of the sources.

        Returns
        -------
        idx: int or None
            Index into mean_nz/sigma_nz/alpha_nz/z_min_tomo_bin, None if no
            tomographic bin satisfies the condition.
        """
        mask = self.z_min_tomo_bin > z_obs + Delta_z
        if not np.any(mask):
            return None
        return int(np.argmax(mask))

    def n_zs_norM(self, idx):
        r"""
        Normalization of the source redshift distribution between 0 and zs_max.

        Parameters
        ----------
        idx: int
            Index into mean_nz/sigma_nz/alpha_nz for the tomographic bin,
            as returned by `get_tomo_bin_index`.

        Returns
        -------
        n_zs_norM: float
            Inverse of the integral of `n_zs` between 0 and zs_max.
        """
        return 1.0 / (
            skewnorm.cdf(
                self.zs_max,
                self.alpha_nz[idx],
                self.mean_nz[idx],
                self.sigma_nz[idx],
            )
            - skewnorm.cdf(
                0.0,
                self.alpha_nz[idx],
                self.mean_nz[idx],
                self.sigma_nz[idx],
            )
        )

    def n_zs(self, z, idx):
        r"""
        Redshift distribution of the sources.

        Parameters
        ----------
        z: float or np.ndarray
            Source redshift.
        idx: int
            Index into mean_nz/sigma_nz/alpha_nz for the tomographic bin,
            as returned by `get_tomo_bin_index`.

        Returns
        -------
        n_zs: float or np.ndarray
            Skew-normal redshift distribution of the sources in the tomographic
            bins >= idx, at the source redshift z (not normalized)
        """
        return skewnorm.pdf(
            z,
            self.alpha_nz[idx],
            self.mean_nz[idx],
            self.sigma_nz[idx],
        )

    def sigma_crit_inv_eff(self, z, Delta_z=0.05, z_grid_size=50):
        r"""
        Effective inverse critical surface mass density.

        Computes the effective inverse critical surface mass density at
        the true cluster redshift:

        ..math:
            \langle \Sigma_{\rm crit}^{-1} \rangle(z) =
            \frac{\int_{z}^{z_{\rm s,max}} dz_s\, n_t(z_s)\, \Sigma_{\rm crit}^{-1}(z, z_s)}
            {\int_{0}^{z_{\rm s,max}} dz_s\, n_t(z_s)}

        where n_t is the redshift distribution of the sources in the tomographic
        bins t, t+1, ... selected with `get_tomo_bin_index`, approximating the
        observed redshift of the cluster with its true redshift. The sources in
        front of the cluster have zero lensing signal. The result is zero if no
        tomographic bin satisfies the condition, or for z >= zs_max.

        Parameters
        ----------
        z: np.ndarray
            True cluster redshift.
        Delta_z : float, optional
            Minimum redshift separation between the clusters and the lower edge of
            the tomographic bins of the sources.
        z_grid_size: int, optional
            Number of source-redshift grid points used for the quadrature

        Returns
        -------
        m_sigma_crit_m1: np.ndarray
            Effective inverse critical surface mass density (units : pc^2 / Msun / h)
        """
        z = np.asarray(z)
        sig_crit_inv_eff = np.zeros(z.shape)

        # first tomographic bin of the sources of each cluster (None: no sources)
        tomo_bins = [self.get_tomo_bin_index(z_cl, Delta_z) for z_cl in z]

        for ind_tomo in sorted({t for t in tomo_bins if t is not None}):
            # clusters with these sources (no sources behind clusters with z >= zs_max)
            clusters = np.array([t == ind_tomo for t in tomo_bins]) & (z < self.zs_max)
            if not np.any(clusters):
                continue
            z_s = np.linspace(z[clusters], self.zs_max, z_grid_size, axis=1)
            # last point slightly below zs_max, where the distances may not be tabulated
            z_s[:, -1] = self.zs_max - 1.0e-5
            with np.errstate(divide="ignore", invalid="ignore"):
                # zero lensing for sources at the cluster redshift
                sig_crit_m1 = self.n_zs(z_s, ind_tomo) / self.sigma_crit(
                    z[clusters], z_s
                )
            sig_crit_inv_eff[clusters] = self.n_zs_norM(ind_tomo) * simpson(
                np.nan_to_num(sig_crit_m1, nan=0.0, posinf=0.0), x=z_s
            )  # pc^2 / Msun / h
        return sig_crit_inv_eff

    def surface_mass_density_args(self, R, z, M, radius_units="Mpc/h"):
        r"""
        Prepare arguments for _model_surface_mass_density_profile and
        _model_mean_surface_mass_density_profile with correct shapes.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        R_outshape: np.ndarray
            Radius (units: Mpc / h) with shape (1, 1, R.size) if radius_units="Mpc/h"
            else (z.size, 1, R.size)
        RDelta: np.ndarray
            Radius of overdensity (units: Mpc / h) with shape (z.size, M.size, 1)
        densityThreshold: np.ndarray
            Threshold density (units : h * Msun / Mpc**2)  with shape (z.size, 1, 1)
        """
        densityThreshold = np.atleast_1d(
            convert_to_Delta_crit(
                self.overdensity_type, self.overdensity, self.background, z
            )
            * derived_cosmology.rho_crit(self.background, z)
            / self.background.h**2.0
        )[:, np.newaxis, np.newaxis]

        RDelta = (
            3.0 * M[np.newaxis, :, np.newaxis] / 4.0 / np.pi / densityThreshold
        ) ** (1.0 / 3.0)

        if radius_units.lower() != "mpc/h":
            D_A = (
                self.halo_model_properties.angular_diameter_distance(z)
                * self.background.h
            )  # Mpc / h
            R_outshape = convert_distance(R, radius_units, "Mpc/h", D_A[:, np.newaxis])[
                :, np.newaxis
            ]
        else:
            R_outshape = R[np.newaxis, np.newaxis, :]

        return R_outshape, RDelta, densityThreshold

    def check_profile_shape(self, R, z, M, profile):
        """
        Check the shape of the mass profile.
        """
        expected_shape = (
            np.atleast_1d(z).size,
            np.atleast_1d(M).size,
            np.atleast_1d(R).size,
        )
        assert_msg = f"Expected shape {expected_shape}, got {profile.shape}"
        assert profile.shape == expected_shape, assert_msg

    def _include_2h_term(
        self, inclusion_type, term_1h, func_2h, R, z, halo_bias, radius_units
    ):
        """
        Computes and adds the 2-halo term.

        Parameters
        ----------
        term_1h : np.ndarray
            1 halo term, with shape (z.size, M.size, R.size).
        func_2h : function
            Function that computes the 2h term. It must take (R, z, radius_units)
            inputs, and output shape (z.size, R.size)
        inclusion_type : str
            If "sum", the 1-halo and 2-halo profile are summed.
            If "max", the maximum between them is considered at each point.
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        """
        if inclusion_type not in ("sum", "max"):
            raise ValueError(
                "Invalid 'inclusion_type' definition, %s." % inclusion_type
            )

        # Check 2h inputs
        if halo_bias is None:
            raise ValueError(
                "halo_bias must be provided explicitly when computing the 2-halo term."
            )

        if halo_bias.shape != term_1h.shape[:2]:
            raise ValueError(
                f"halo_bias shape {halo_bias.shape} must "
                f"be the same as first two of term_1h {term_1h.shape}"
            )

        # compute and add 2h term
        term_2h = (
            func_2h(R, z, radius_units)[:, np.newaxis, :] * halo_bias[:, :, np.newaxis]
        )

        if inclusion_type == "sum":
            return term_1h + term_2h
        elif inclusion_type == "max":
            return np.maximum(term_1h, term_2h)

    def include_surface_mass_density_2h(
        self, Sigma_1h, inclusion_type, R, z, halo_bias, radius_units="Mpc/h"
    ):
        r"""
        Include the contribution of the cosmological 2-halo term.

        Include the contribution of the 2-halo term to the surface density profile.

        Parameters
        ----------
        Sigma_1h : np.ndarray
            One-halo surface mass density.
        inclusion_type : str
            If "sum", the 1-halo and 2-halo profile are summed.
            If "max", the maximum between them is considered at each point.
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            Total surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        return self._include_2h_term(
            inclusion_type,
            Sigma_1h,
            func_2h=self.halo_model_properties.surface_mass_density_2h,
            R=R,
            z=z,
            halo_bias=halo_bias,
            radius_units=radius_units,
        )

    def include_excess_surface_mass_density_2h(
        self, DeltaSigma_1h, inclusion_type, R, z, halo_bias, radius_units="Mpc/h"
    ):
        r"""
        Include the contribution of the cosmological 2-halo term.

        Include the contribution of the 2-halo term to the excess surface density profile.

        Parameters
        ----------
        DeltaSigma_1h : np.ndarray
            One-halo surface mass density.
        inclusion_type : str
            If "sum", the 1-halo and 2-halo profile are summed.
            If "max", the maximum between them is considered at each point.
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        DeltaSigma: np.ndarray
            Total excess surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        return self._include_2h_term(
            inclusion_type,
            DeltaSigma_1h,
            func_2h=self.halo_model_properties.excess_surface_mass_density_2h,
            R=R,
            z=z,
            halo_bias=halo_bias,
            radius_units=radius_units,
        )
