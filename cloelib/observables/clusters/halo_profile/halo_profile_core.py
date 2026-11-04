import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.special import i0e
from scipy.stats import rayleigh, skewnorm

from cloelib.auxiliary import units
from cloelib.cosmology import derived_cosmology
from cloelib.observables.clusters.auxiliary import (
    convert_distance,
    convert_to_Delta_crit,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics


def _piecewise_unit(x, below, at_one, above):
    r"""
    Evaluate a function of the dimensionless radius x defined separately for x < 1,
    x = 1 and x > 1, element-wise on arrays (scalar in, scalar out).
    """
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, at_one)
    lo, hi = x < 1.0, x > 1.0
    out[lo] = below(x[lo])
    out[hi] = above(x[hi])
    return out if out.ndim else float(out)


class HaloProfileCore:
    def __init__(
        self,
        matter_statistics: MatterStatistics,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        zs_max: float = 4.0,
        mean_nz: np.ndarray = None,
        sigma_nz: np.ndarray = None,
        alpha_nz: np.ndarray = None,
        z_mean_tomo_bin: np.ndarray = None,
        exclusion_background=1.0,
    ):
        r"""Auxiliary class computing quantities used in mass profile models.

        Initialize the class with given perturbations and overdensity definition.

        Parameters
        ----------
        matter_statistics : MatterStatistics
            An object from the `MatterStatistics` class.
        mean_nz : np.ndarray
            Mean redshift per tomographic bin, shape (Number of tomographic source redshift bins,).
        sigma_nz : np.ndarray
            Redshift scatter per tomographic bin, shape (Number of tomographic source redshift bins,).
        alpha_nz : np.ndarray
            Skewness parameter per tomographic bin, shape (Number of tomographic source redshift bins,).
        z_mean_tomo_bin : np.ndarray
            Mean z_obs boundary per tomographic bin, shape (Number of tomographic source redshift bins,),
            used to select which tomo bin a given mean_z_obs_bin falls into.
        exclusion_background : float or callable, optional
            Fraction of the mean matter density excluded inside the exclusion radius
            by the 2-halo term with halo exclusion (`two_halo="sum_exclusion"`),
            a number or a function of the redshift. Default 1 (all the matter);
            e.g. for a mock where only the halos above a minimum mass are painted,
            the mass fraction in those halos.
        """
        self.matter_statistics = matter_statistics
        self.exclusion_background = exclusion_background
        self._exclusion_cache = {}
        self._rice_cache = {}
        self.overdensity_type = overdensity_type
        self.overdensity = overdensity
        self.zs_max = zs_max
        self.mean_nz = np.atleast_1d(
            mean_nz if mean_nz is not None else np.array([0.18739829, 0.33775482, 0.50338379, 0.66827123, 0.86322198,
       1.30780699])
        )
        self.sigma_nz = np.atleast_1d(
            sigma_nz if sigma_nz is not None else np.array([0.85102341, 0.77623404, 0.74479123, 0.79814509, 0.85482975,
       0.77782535])
        )
        self.alpha_nz = np.atleast_1d(
            alpha_nz if alpha_nz is not None else np.array([7.4511585 , 3.12760121, 3.24293517, 6.36498593, 4.32074969,
       1.50975262])
        )
        self.z_mean_tomo_bin = np.atleast_1d(
            z_mean_tomo_bin if z_mean_tomo_bin is not None else np.array([0.52372988, 0.5987506 , 0.7179003 , 0.90270588, 1.19243611,
       1.71458837])
        )

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.matter_statistics.background

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
        d_a_sources = self.matter_statistics.angular_diameter_distance(z_sources)  # Mpc
        d_m_sources = (1.0 + z_sources) * d_a_sources
        d_a_lens = self.matter_statistics.angular_diameter_distance(z)[
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

    def _get_tomo_bin_index(self, mean_z_obs_bin):
        r"""
        Select the tomographic bin index for a given mean_z_obs_bin.

        Returns the first index i such that z_mean_tomo_bin[i] > mean_z_obs_bin.

        Parameters
        ----------
        mean_z_obs_bin : float
            Mean redshift of the observed redshift bin considered.

        Returns
        -------
        idx: int
            Index into mean_nz/sigma_nz/alpha_nz/z_mean_tomo_bin.
        """
        mask = self.z_mean_tomo_bin > mean_z_obs_bin
        if not np.any(mask):
            raise ValueError(
                f"mean_z_obs_bin={mean_z_obs_bin} is not below any value in "
                f"z_mean_tomo_bin={self.z_mean_tomo_bin}"
            )
        return np.argmax(mask)

    def n_zs_norM(self, z, idx, Delta_z=0.05):
        r"""
        Galaxy number density normalization.

        Parameters
        ----------
        z: float or np.ndarray
            Lens redshift.
        idx: int
            Index into mean_nz/sigma_nz/alpha_nz for the tomographic bin,
            as returned by `_get_tomo_bin_index`.
        Delta_z: float, optional
            Redshift buffer defining the lower edge of the source integration
            range, z_s > z + Delta_z. Sources within Delta_z of the cluster
            are excluded to avoid contamination from cluster-member/foreground
            galaxies scattered into the source sample.
        Returns
        -------
        n_zs_norM: float or np.ndarray
            Galaxy number density normalization per lens redshift and z_obs bin.
        """
        n_zs_norM = 1.0 / (
            skewnorm.cdf(
                self.zs_max,
                self.alpha_nz[idx],
                self.mean_nz[idx],
                self.sigma_nz[idx],
            )
            - skewnorm.cdf(
                z+ Delta_z,
                self.alpha_nz[idx],
                self.mean_nz[idx],
                self.sigma_nz[idx],
            )
        )

        return n_zs_norM

    def n_zs(self, z, idx):
        r"""
        Galaxy number density.

        Parameters
        ----------
        z: float or np.ndarray
            Lens redshift.
        idx: int
            Index into mean_nz/sigma_nz/alpha_nz for the tomographic bin,
            as returned by `_get_tomo_bin_index`.

        Returns
        -------
        n_zs: float or np.ndarray
            Galaxy number density per source redshift for a given z_obs bin
        """
        return skewnorm.pdf(
            z,
            self.alpha_nz[idx],
            self.mean_nz[idx],
            self.sigma_nz[idx],
        )

    def sigma_crit_inv_eff(self, z, idx, Delta_z = 0.05, z_grid_size=50, power=1):
        r"""
        Effective inverse critical surface mass density.

        Computes the effective critical surface mass density at
        the true cluster redshift.

        Parameters
        ----------
        z: float or np.ndarray
            true cluster redshift.
        idx: int
            Index into mean_nz/sigma_nz/alpha_nz for the tomographic bin,
            as returned by `_get_tomo_bin_index`.
        Delta_z: float, optional
            Redshift buffer defining the lower edge of the source integration
            range, z_s > z + Delta_z. Sources within Delta_z of the cluster
            are excluded to avoid contamination from cluster-member/foreground
            galaxies scattered into the source sample.
        z_grid_size: int, optional
            Number of source-redshift grid points used for the quadrature
        power: int, optional
            Power n of the inverse critical surface mass density averaged over the
            sources, <Sigma_crit^-n>. Default 1; n=2 is used for the reduced shear.
        Returns
        -------
        m_sigma_crit_m1: float
            Effective inverse critical surface mass density to the power n
            (units : (pc^2 / Msun / h)^n)
        """
        z_s = np.linspace(z + Delta_z, self.zs_max, z_grid_size, axis=1) 
        z_s[z_s>=self.zs_max]=self.zs_max-1.0e-5 # the last term is to avoid problem with the normalization n_zs_norM. It can be removed setting zs_max higer than max z (i.e. z_true)
        sig_crit_m1 = self.n_zs(z_s,idx) * 1.0 / self.sigma_crit(z, z_s)**power
        with np.errstate(divide="ignore", invalid="ignore"):
            sig_crit_inv_eff = self.n_zs_norM(z, idx, Delta_z) * simpson(
                sig_crit_m1, x=z_s
            )  # pc^2 / Msun / h
        # no sources behind clusters with z + Delta_z >= zs_max
        return np.where(np.asarray(z) + Delta_z < self.zs_max, sig_crit_inv_eff, 0.0)


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
                self.matter_statistics.angular_diameter_distance(z) * self.background.h
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
        assert (
            profile.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {profile.shape}"

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
        self,
        Sigma_1h,
        inclusion_type,
        R,
        z,
        halo_bias,
        radius_units="Mpc/h",
        sigma_off=None,
        R_exc=None,
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
            If "sum_exclusion", the 1-halo and the 2-halo profile with nonlinear
            power spectrum and halo exclusion inside R_exc are summed
            (see `two_halo_exclusion`).
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        sigma_off: float, optional
            Scale of the Rayleigh miscentering distribution (units : Mpc / h).
            If given, the miscentered 2-halo term is computed with the exact
            average over the miscentering distribution (see
            `_miscentered_2h_conv`), otherwise the centered one is used.
        R_exc: np.ndarray, optional
            Exclusion radius (units : Mpc / h), with shape (z.size, M.size).
            Required by inclusion_type="sum_exclusion".

        Returns
        -------
        Sigma: np.ndarray
            Total surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        if inclusion_type == "sum_exclusion":
            return Sigma_1h + self.two_halo_exclusion(
                R, z, R_exc, halo_bias, sigma_off=sigma_off, excess=False, radius_units=radius_units
            )
        if sigma_off is None:
            func_2h = self.matter_statistics.surface_mass_density_2h
        else:

            def func_2h(R, z, radius_units):
                return self.surface_mass_density_2h_off(
                    R, z, sigma_off, radius_units=radius_units
                )

        return self._include_2h_term(
            inclusion_type,
            Sigma_1h,
            func_2h=func_2h,
            R=R,
            z=z,
            halo_bias=halo_bias,
            radius_units=radius_units,
        )

    def include_excess_surface_mass_density_2h(
        self,
        DeltaSigma_1h,
        inclusion_type,
        R,
        z,
        halo_bias,
        radius_units="Mpc/h",
        sigma_off=None,
        R_exc=None,
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
            If "sum_exclusion", the 1-halo and the 2-halo profile with nonlinear
            power spectrum and halo exclusion inside R_exc are summed
            (see `two_halo_exclusion`).
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        sigma_off: float, optional
            Scale of the Rayleigh miscentering distribution (units : Mpc / h).
            If given, the miscentered 2-halo term is computed with the exact
            average over the miscentering distribution (see
            `_miscentered_2h_conv`), otherwise the centered one is used.
        R_exc: np.ndarray, optional
            Exclusion radius (units : Mpc / h), with shape (z.size, M.size).
            Required by inclusion_type="sum_exclusion".

        Returns
        -------
        DeltaSigma: np.ndarray
            Total excess surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        if inclusion_type == "sum_exclusion":
            return DeltaSigma_1h + self.two_halo_exclusion(
                R, z, R_exc, halo_bias, sigma_off=sigma_off, excess=True, radius_units=radius_units
            )
        if sigma_off is None:
            func_2h = self.matter_statistics.excess_surface_mass_density_2h
        else:

            def func_2h(R, z, radius_units):
                return self.excess_surface_mass_density_2h_off(
                    R, z, sigma_off, radius_units=radius_units
                )

        return self._include_2h_term(
            inclusion_type,
            DeltaSigma_1h,
            func_2h=func_2h,
            R=R,
            z=z,
            halo_bias=halo_bias,
            radius_units=radius_units,
        )

    def _miscentered_2h_conv(
        self,
        R,
        z,
        sigma_off,
        excess=False,
        radius_units="Mpc/h",
        R_min=1.0e-3,
        dlnR=0.02,
        n_sigma_id=25.0,
    ):
        r"""
        Miscentered 2-halo term from the exact average over the offsets.

        For a Rayleigh distribution of the offsets with scale `sigma_off`
        (2D Gaussian offset), the azimuthally averaged surface density around the
        offset centre is the convolution of the centered profile with the Rice
        distribution of the distance from the true centre:

        .. math::

            \Sigma_{\rm off}(R) = \int_0^\infty dR'\, \frac{R'}{\sigma_{\rm off}^2}
            e^{-(R^2 + R'^2)/2\sigma_{\rm off}^2}\, I_0\left(\frac{R R'}
            {\sigma_{\rm off}^2}\right) \Sigma_{\rm 2h}(R')

        The centered 2-halo surface density is computed once on a logarithmic grid
        and the kernel is applied with a matrix product. The excess surface density
        is :math:`\Delta\Sigma_{\rm off}(R) = \bar\Sigma_{\rm off}(<R) -
        \Sigma_{\rm off}(R)`, i.e. the tangential shear around the offset centre.
        For :math:`R > n_{\sigma}\,\sigma_{\rm off}` the miscentering correction is
        of order :math:`(\sigma_{\rm off}/R)^2` and the centered profile is used.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        sigma_off: float
            Scale of the Rayleigh miscentering distribution (units : Mpc / h).
        excess: bool, optional
            If True, return the excess surface density, otherwise the surface density.
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        R_min: float, optional
            Minimum radius of the integration grid (units : Mpc / h).
        dlnR: float, optional
            Logarithmic spacing of the integration grid.
        n_sigma_id: float, optional
            The centered profile is used for R > n_sigma_id * sigma_off.

        Returns
        -------
        profile: np.ndarray
            Miscentered 2-halo surface or excess surface density profile
            (units : h * Msun / pc**2). Shape: (z.size, R.size).
        """
        z = np.atleast_1d(z)

        # radius in Mpc/h with shape (z.size, R.size)
        if radius_units.lower() != "mpc/h":
            D_A = (
                self.matter_statistics.angular_diameter_distance(z) * self.background.h
            )  # Mpc / h
            R_mpc = convert_distance(R, radius_units, "Mpc/h", D_A[:, np.newaxis])
        else:
            R_mpc = np.broadcast_to(np.atleast_1d(R), (z.size, np.size(R)))

        # integration grid and centered 2-halo surface density, shape (z.size, N)
        R_id = n_sigma_id * sigma_off
        R_max = max(R_mpc.max(), R_id + 8.0 * sigma_off)
        R_grid = np.exp(np.arange(np.log(R_min), np.log(R_max) + dlnR, dlnR))
        Sigma = self.matter_statistics.surface_mass_density_2h(R_grid, z, "Mpc/h")

        n_ker, inner, kernel = self._cached_rice_kernel(R_grid, sigma_off, R_id, dlnR)

        Sigma_off = Sigma.copy()
        Sigma_off[:, :n_ker][:, inner] = Sigma[:, :n_ker] @ kernel.T

        if excess:
            # mean surface density within R (constant profile below R_min)
            cumulative = (
                cumulative_trapezoid(
                    Sigma_off * R_grid**2.0, np.log(R_grid), axis=1, initial=0.0
                )
                + 0.5 * Sigma_off[:, :1] * R_min**2.0
            )
            profile_grid = 2.0 * cumulative / R_grid**2.0 - Sigma_off
        else:
            profile_grid = Sigma_off

        return np.array(
            [
                np.interp(np.log(R_mpc[i]), np.log(R_grid), profile_grid[i])
                for i in range(z.size)
            ]
        )

    def _cached_rice_kernel(self, R_grid, sigma_off, R_id, dlnR):
        r"""
        Rice kernel of `_rice_kernel` on the first `n_ker` points of R_grid, cached.

        The kernel of the radii R < R_id only involves R' < R_id + 8 sigma_off, i.e. the
        first `n_ker` points of the logarithmic grid, and it does not depend on z.

        Returns
        -------
        n_ker: int
            Number of points of R_grid used by the kernel.
        inner, kernel: np.ndarray
            Output of `_rice_kernel` on R_grid[:n_ker].
        """
        n_ker = int(np.searchsorted(R_grid, R_id + 8.0 * sigma_off, side="right"))
        key = (float(sigma_off), float(R_id), float(R_grid[0]), float(dlnR), n_ker)
        if key not in self._rice_cache:
            if len(self._rice_cache) > 64:
                self._rice_cache.clear()
            self._rice_cache[key] = self._rice_kernel(R_grid[:n_ker], sigma_off, R_id, dlnR)
        return (n_ker,) + self._rice_cache[key]

    @staticmethod
    def _rice_kernel(R_grid, sigma_off, R_id, dlnR):
        r"""
        Rice kernel of the Rayleigh miscentering for the radii R_grid < R_id.

        Parameters
        ----------
        R_grid: np.ndarray
            Logarithmic radial grid (units : Mpc / h), with spacing dlnR.
        sigma_off: float
            Scale of the Rayleigh miscentering distribution (units : Mpc / h).
        R_id: float
            The kernel is computed for R_grid < R_id.
        dlnR: float
            Logarithmic spacing of R_grid.

        Returns
        -------
        inner: np.ndarray
            Mask of the radii R_grid < R_id.
        kernel: np.ndarray
            Kernel with the integration weights (dR' = R' dlnR'), normalized to
            unit integral, shape (inner.sum(), R_grid.size).
        """
        # i0e(t) = exp(-t) I0(t): exp[-(x^2+y^2)/2s^2] I0(xy/s^2) = exp[-(x-y)^2/2s^2] i0e(xy/s^2)
        inner = R_grid < R_id
        x, y = R_grid[inner, np.newaxis], R_grid[np.newaxis, :]
        kernel = (
            y**2.0
            / sigma_off**2.0
            * np.exp(-0.5 * (x - y) ** 2.0 / sigma_off**2.0)
            * i0e(x * y / sigma_off**2.0)
        )
        weights = np.full(R_grid.size, dlnR)
        weights[[0, -1]] *= 0.5
        kernel *= weights[np.newaxis, :]
        kernel /= kernel.sum(axis=1)[:, np.newaxis]  # int K dR' = 1
        return inner, kernel

    def _exclusion_background_fraction(self, z):
        """Fraction of the mean density excluded by the 2-halo term with exclusion, shape (z.size,)"""
        if callable(self.exclusion_background):
            return np.broadcast_to(np.asarray(self.exclusion_background(z), dtype=float), z.shape)
        return np.full(z.shape, float(self.exclusion_background))

    def _two_halo_exclusion_grid(self, z, R_exc, halo_bias, R_grid, n_t=200, t_max=50.0, n_y=201):
        r"""
        Centered 2-halo surface density with halo exclusion on a radial grid.

        See `two_halo_exclusion`. The result is cached, so that the surface and
        excess surface densities, and the miscentered populations, share it.

        Parameters
        ----------
        z: np.ndarray
            Redshift, shape (z.size,).
        R_exc: np.ndarray
            Exclusion radius (units : Mpc / h), shape (z.size, M.size).
        halo_bias: np.ndarray
            Halo bias, shape (z.size, M.size).
        R_grid: np.ndarray
            Radial grid (units : Mpc / h).
        n_t: int, optional
            Number of points of the integral along the line of sight.
        t_max: float, optional
            Maximum of the integral along the line of sight, in units of R_exc.
        n_y: int, optional
            Number of points of the table of the line-of-sight integral D in
            y = pi_e / R_exc (linear interpolation).

        Returns
        -------
        Sigma: np.ndarray
            Surface density (units : h * Msun / pc**2), shape (z.size, M.size, R_grid.size).
        """
        key = (z.tobytes(), R_exc.tobytes(), halo_bias.tobytes(), R_grid[0], R_grid[-1], R_grid.size)
        if key in self._exclusion_cache:
            return self._exclusion_cache[key]

        ms = self.matter_statistics
        # unbiased 2-halo surface density with the nonlinear power spectrum, (z.size, N)
        Sigma_nl = ms.surface_mass_density_2h(R_grid, z, "Mpc/h", nonlinear=True)
        rho_m = (
            self.background.Omega_m(z) * derived_cosmology.rho_crit(self.background, z) / self.background.h**2
        )  # h^2 Msun / Mpc^3
        f_bg = self._exclusion_background_fraction(z)
        # line-of-sight distance from the exclusion sphere, in units of R_exc (t = 0 included)
        u = np.concatenate([[0.0], np.geomspace(1.0e-4, t_max, n_t)])
        # trapezoidal weights in u (t = R_exc u)
        w_u = np.zeros(u.size)
        w_u[1:] += 0.5 * np.diff(u)
        w_u[:-1] += 0.5 * np.diff(u)
        # D(R) = R_exc int_0^inf xi(R_exc sqrt(1 + u^2)) [u / sqrt(u^2 + y^2) - 1] du only depends
        # on y = pi_e / R_exc through the weight: kernel on a grid of y in [0, 1], (Y, U)
        y_tab = np.linspace(0.0, 1.0, n_y)
        norm = np.sqrt(u[np.newaxis, :] ** 2 + y_tab[:, np.newaxis] ** 2)
        kernel_y = (u[np.newaxis, :] / np.maximum(norm, 1.0e-30) - 1.0) * w_u[np.newaxis, :]
        kernel_y[0] = 0.0  # y = 0: R = R_exc
        Sigma = np.empty((z.size,) + R_exc.shape[1:] + (R_grid.size,))
        for i in range(z.size):
            b, r_exc = halo_bias[i], R_exc[i]
            Sigma_nl_exc = np.interp(np.log(r_exc), np.log(R_grid), Sigma_nl[i])  # (M,)
            t = r_exc[:, np.newaxis] * u[np.newaxis, :]  # (M, U)
            # comoving separation for the comoving correlation function
            r_com = (1.0 + z[i]) * np.sqrt(r_exc[:, np.newaxis] ** 2 + t**2)
            xi = ms.correlation_function_cb_nl(r_com.ravel(), z[i : i + 1])[0].reshape(r_com.shape)
            inner = R_grid < r_exc.max()
            pi_e2 = np.clip(r_exc[:, np.newaxis] ** 2 - R_grid[np.newaxis, inner] ** 2, 0.0, None)  # (M, N_in)
            # D(R) = int_0^inf xi(sqrt(R_exc^2 + t^2)) [t / sqrt(t^2 + pi_e^2) - 1] dt, tabulated
            # in y = pi_e / R_exc, (M, Y), and linearly interpolated (D is not used outside R_exc)
            D_tab = (xi * r_exc[:, np.newaxis]) @ kernel_y.T
            x = np.sqrt(pi_e2) / r_exc[:, np.newaxis] * (n_y - 1)
            j = np.minimum(x.astype(int), n_y - 2)
            frac = x - j
            D = (1.0 - frac) * np.take_along_axis(D_tab, j, axis=1) + frac * np.take_along_axis(D_tab, j + 1, axis=1)
            # inside R_exc: b Sigma_NL(R_exc) + 2 rho_m [b D(R) - f_bg pi_e]
            Sigma_in = b[:, np.newaxis] * Sigma_nl_exc[:, np.newaxis] + 2.0e-12 * rho_m[i] * (
                b[:, np.newaxis] * D - f_bg[i] * np.sqrt(pi_e2)
            )
            Sigma_i = b[:, np.newaxis] * Sigma_nl[i][np.newaxis, :]
            excluded = R_grid[np.newaxis, inner] < r_exc[:, np.newaxis]
            Sigma_i[:, inner] = np.where(excluded, Sigma_in, Sigma_i[:, inner])
            Sigma[i] = Sigma_i

        if len(self._exclusion_cache) > 64:
            self._exclusion_cache.clear()
        self._exclusion_cache[key] = Sigma
        return Sigma

    def two_halo_exclusion(
        self,
        R,
        z,
        R_exc,
        halo_bias,
        sigma_off=None,
        excess=False,
        radius_units="Mpc/h",
        R_min=1.0e-3,
        R_max=30.0,
        dlnR=0.02,
        n_sigma_id=25.0,
    ):
        r"""
        2-halo term with nonlinear power spectrum and halo exclusion.

        The matter around a halo is distributed as :math:`\bar\rho_m [1 + b\,
        \xi_{\rm NL}(r)]` outside the exclusion radius :math:`R_{\rm exc}` (e.g.
        :math:`R_\Delta`) and is absent inside it (hard exclusion), where also a
        fraction `exclusion_background` of the mean density is removed. The
        projected excess density is, for :math:`R < R_{\rm exc}`,

        .. math::

            \Sigma_{\rm 2h}(R) = 2\bar\rho_m \left[b \int_{\pi_e}^\infty
            \xi_{\rm NL}\left(\sqrt{R^2 + \pi^2}\right) d\pi - f_{\rm bg}\,\pi_e
            \right], \qquad \pi_e = \sqrt{R_{\rm exc}^2 - R^2},

        computed as :math:`b\,\Sigma_{\rm NL}(R_{\rm exc}) + 2\bar\rho_m [b\,
        D(R) - f_{\rm bg}\pi_e]` with :math:`D(R) = \int_0^\infty
        \xi_{\rm NL}(\sqrt{R_{\rm exc}^2 + t^2})\,[t/\sqrt{t^2 + \pi_e^2} - 1]\,dt`,
        which needs the correlation function only outside the exclusion radius
        (the small scales cancel). For :math:`R \ge R_{\rm exc}`,
        :math:`\Sigma_{\rm 2h} = b\,\Sigma_{\rm NL}(R)`, the projection of the
        nonlinear power spectrum. The excess surface density and the miscentering
        (exact average over a Rayleigh distribution, as in `_miscentered_2h_conv`)
        are computed on a logarithmic radial grid.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        R_exc: np.ndarray
            Exclusion radius (units : Mpc / h), shape (z.size, M.size).
        halo_bias: np.ndarray
            Halo bias, shape (z.size, M.size).
        sigma_off: float, optional
            Scale of the Rayleigh miscentering distribution (units : Mpc / h);
            centered profile if None.
        excess: bool, optional
            If True, return the excess surface density, otherwise the surface density.
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        R_min, R_max: float, optional
            Range of the radial grid (units : Mpc / h); R_max is extended to the
            largest R and to the miscentering kernel if needed.
        dlnR: float, optional
            Logarithmic spacing of the radial grid.
        n_sigma_id: float, optional
            The centered profile is used for R > n_sigma_id * sigma_off.

        Returns
        -------
        profile: np.ndarray
            2-halo surface or excess surface density (units : h * Msun / pc**2),
            shape (z.size, M.size, R.size).
        """
        z = np.atleast_1d(z)
        if R_exc is None or halo_bias is None:
            raise ValueError("The 2-halo term with exclusion requires R_exc and halo_bias.")
        R_exc = np.asarray(R_exc, dtype=float).reshape(z.size, -1)
        halo_bias = np.asarray(halo_bias, dtype=float).reshape(R_exc.shape)

        # radius in Mpc/h with shape (z.size, R.size)
        if radius_units.lower() != "mpc/h":
            D_A = self.matter_statistics.angular_diameter_distance(z) * self.background.h  # Mpc / h
            R_mpc = convert_distance(R, radius_units, "Mpc/h", D_A[:, np.newaxis])
        else:
            R_mpc = np.broadcast_to(np.atleast_1d(R), (z.size, np.size(R)))

        R_max = max(R_max, R_mpc.max())
        if sigma_off is not None:
            R_id = n_sigma_id * sigma_off
            R_max = max(R_max, R_id + 8.0 * sigma_off)
        R_grid = np.exp(np.arange(np.log(R_min), np.log(R_max) + dlnR, dlnR))
        profile_grid = self._two_halo_exclusion_grid(z, R_exc, halo_bias, R_grid)

        if sigma_off is not None:
            n_ker, inner, kernel = self._cached_rice_kernel(R_grid, sigma_off, R_id, dlnR)
            profile_grid = profile_grid.copy()
            profile_grid[..., :n_ker][..., inner] = profile_grid[..., :n_ker] @ kernel.T

        if excess:
            # mean surface density within R (constant profile below R_min)
            cumulative = (
                cumulative_trapezoid(profile_grid * R_grid**2.0, np.log(R_grid), axis=-1, initial=0.0)
                + 0.5 * profile_grid[..., :1] * R_min**2.0
            )
            profile_grid = 2.0 * cumulative / R_grid**2.0 - profile_grid

        return np.array(
            [
                [np.interp(np.log(R_mpc[i]), np.log(R_grid), p) for p in profile_grid[i]]
                for i in range(z.size)
            ]
        )

    def surface_mass_density_2h_off(self, R, z, sigma_off, radius_units="Mpc/h"):
        r"""
        Miscentered surface 2-halo matter density profile.

        Computes the cosmological unbiased surface 2-halo density profile at
        radius R, averaged over a Rayleigh miscentering distribution
        (see `_miscentered_2h_conv`).

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        sigma_off: float
            Scale of the Rayleigh miscentering distribution (units : Mpc / h).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            Miscentered 2-halo surface mass density profile
            (units : h * Msun / pc**2). Shape: (z.size, R.size).
        """
        return self._miscentered_2h_conv(
            R, z, sigma_off, excess=False, radius_units=radius_units
        )

    def excess_surface_mass_density_2h_off(
        self, R, z, sigma_off, radius_units="Mpc/h"
    ):
        r"""
        Miscentered excess surface 2-halo matter density profile.

        Computes the cosmological unbiased excess surface 2-halo density
        profile at radius R around the offset centre, averaged over a Rayleigh
        miscentering distribution (see `_miscentered_2h_conv`).

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        sigma_off: float
            Scale of the Rayleigh miscentering distribution (units : Mpc / h).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        DeltaSigma: np.ndarray
            Miscentered 2-halo excess surface mass density profile
            (units : h * Msun / pc**2). Shape: (z.size, R.size).
        """
        return self._miscentered_2h_conv(
            R, z, sigma_off, excess=True, radius_units=radius_units
        )

    def _miscentered_2h_semiapprox(
        self, func_2h, R, z, sigma_off, radius_units="Mpc/h", roff_grid_size=100
    ):
        r"""
        Miscentered 2-halo term in the semi-approximated form.

        For a fixed miscentering offset :math:`R_{\rm off}`, the azimuthal
        average of the 2-halo profile around the offset centre is approximated as
        :math:`f(\max(R, R_{\rm off}))`, with :math:`f` the centered 2-halo
        profile. The result is then averaged over the Rayleigh miscentering
        distribution of scale `sigma_off`, integrating in :math:`\ln R_{\rm off}`
        between its 0.01% and 99.99% quantiles.

        Parameters
        ----------
        func_2h : function
            Function that computes the centered 2h term. It must take
            (R, z, radius_units) inputs, and output shape (z.size, R.size)
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        sigma_off: float
            Scale of the Rayleigh miscentering distribution (units : Mpc / h).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        roff_grid_size: int, optional
            Number of grid points used for the integration over the
            miscentering offset.

        Returns
        -------
        profile: np.ndarray
            Miscentered 2-halo profile (units : h * Msun / pc**2).
            Shape: (z.size, R.size).
        """
        z = np.atleast_1d(z)

        # radius in Mpc/h with shape (z.size, R.size)
        if radius_units.lower() != "mpc/h":
            D_A = (
                self.matter_statistics.angular_diameter_distance(z) * self.background.h
            )  # Mpc / h
            R_mpc = convert_distance(R, radius_units, "Mpc/h", D_A[:, np.newaxis])
        else:
            R_mpc = np.broadcast_to(np.atleast_1d(R), (z.size, np.size(R)))

        # miscentering offset grid, log-spaced
        ln_roff = np.linspace(
            np.log(rayleigh.ppf(1.0e-4, scale=sigma_off)),
            np.log(rayleigh.ppf(1.0 - 1.0e-4, scale=sigma_off)),
            roff_grid_size,
        )
        roff = np.exp(ln_roff)

        # evaluate the centered 2h term at R and at R_off with a single call
        R_eval = np.concatenate(
            [R_mpc, np.broadcast_to(roff, (z.size, roff.size))], axis=1
        )
        profile_eval = func_2h(R_eval, z, "Mpc/h")
        profile_R = profile_eval[:, : R_mpc.shape[1]]
        profile_roff = profile_eval[:, R_mpc.shape[1] :]

        # profile around each offset centre, shape (z.size, roff.size, R.size)
        profile_off = np.where(
            R_mpc[:, np.newaxis, :] > roff[np.newaxis, :, np.newaxis],
            profile_R[:, np.newaxis, :],
            profile_roff[:, :, np.newaxis],
        )

        # average over the Rayleigh distribution, dR_off = R_off dlnR_off
        weights = rayleigh.pdf(roff, scale=sigma_off) * roff
        return np.trapezoid(
            profile_off * weights[np.newaxis, :, np.newaxis], x=ln_roff, axis=1
        )

    def surface_mass_density_2h_off_semiapprox(
        self, R, z, sigma_off, radius_units="Mpc/h", roff_grid_size=100
    ):
        r"""
        Miscentered surface 2-halo matter density profile.

        Computes the cosmological unbiased surface 2-halo density profile at
        radius R, averaged over a Rayleigh miscentering distribution in the
        semi-approximated form (see `_miscentered_2h_semiapprox`).

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        sigma_off: float
            Scale of the Rayleigh miscentering distribution (units : Mpc / h).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        roff_grid_size: int, optional
            Number of grid points used for the integration over the
            miscentering offset.

        Returns
        -------
        Sigma: np.ndarray
            Miscentered 2-halo surface mass density profile
            (units : h * Msun / pc**2). Shape: (z.size, R.size).
        """
        return self._miscentered_2h_semiapprox(
            self.matter_statistics.surface_mass_density_2h,
            R,
            z,
            sigma_off,
            radius_units=radius_units,
            roff_grid_size=roff_grid_size,
        )

    def excess_surface_mass_density_2h_off_semiapprox(
        self, R, z, sigma_off, radius_units="Mpc/h", roff_grid_size=100
    ):
        r"""
        Miscentered excess surface 2-halo matter density profile.

        Computes the cosmological unbiased excess surface 2-halo density
        profile at radius R, averaged over a Rayleigh miscentering distribution
        in the semi-approximated form (see `_miscentered_2h_semiapprox`).

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        sigma_off: float
            Scale of the Rayleigh miscentering distribution (units : Mpc / h).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        roff_grid_size: int, optional
            Number of grid points used for the integration over the
            miscentering offset.

        Returns
        -------
        DeltaSigma: np.ndarray
            Miscentered 2-halo excess surface mass density profile
            (units : h * Msun / pc**2). Shape: (z.size, R.size).
        """
        return self._miscentered_2h_semiapprox(
            self.matter_statistics.excess_surface_mass_density_2h,
            R,
            z,
            sigma_off,
            radius_units=radius_units,
            roff_grid_size=roff_grid_size,
        )
