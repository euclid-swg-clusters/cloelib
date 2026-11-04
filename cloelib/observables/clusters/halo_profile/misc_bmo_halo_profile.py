import numpy as np
from scipy.integrate import cumulative_trapezoid

from cloelib.observables.clusters.matter_statistics import MatterStatistics

from .bmo_halo_profile import BMOHaloProfile
from .halo_profile_core import HaloProfileCore

# linear operators (centred profile on the grid) -> (Sigma_off, DeltaSigma_off at R), shared
# by all the instances (they do not depend on the cosmology)
_MISCENTERING_OPERATORS = {}


class MiscBMOHaloProfile:
    r"""
    Miscentered BMO halo profile from the exact average over a Rayleigh distribution
    of the offsets.

    Same interface as :class:`MiscBMOHaloProfileEmu`, which it can replace (e.g. as the
    ``miscentered_profile`` of ``ClusterWeakLensing``). For a Rayleigh distribution of
    the offsets with scale :math:`\sigma_\mathrm{off}` (Eq. 8 of `Johnston et al. 2007
    <https://arxiv.org/pdf/0709.1159.pdf>`_), the azimuthally averaged surface density
    around the offset centre is the convolution of the centred BMO profile
    (`Baltz et al. 2009
    <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_) with the Rice
    distribution of the distance from the true centre:

    .. math::

        \Sigma_\mathrm{off}(R) = \int_0^\infty dR'\, \frac{R'}{\sigma_\mathrm{off}^2}
        e^{-(R^2 + R'^2)/2\sigma_\mathrm{off}^2}\, I_0\left(\frac{R R'}
        {\sigma_\mathrm{off}^2}\right) \Sigma(R')

    and :math:`\Delta\Sigma_\mathrm{off}(R) = \bar\Sigma_\mathrm{off}(<R) -
    \Sigma_\mathrm{off}(R)`. The centred profile is evaluated on a logarithmic grid
    of radii, and the kernel, the mean within R and the interpolation at the requested
    radii are applied as one linear operator, cached for each (sigma_off, R).
    The 2-halo term is the same as in :class:`MiscBMOHaloProfileEmu`.

    Compared with a brute-force average over the offsets and the angles, the accuracy
    is ~1e-4 for :math:`R \gtrsim \sigma_\mathrm{off}` (~1e-3 at
    :math:`R \sim \sigma_\mathrm{off}`) with the default grid, about ten times better
    than the emulator, and it is faster (no network evaluation).

    Parameters
    ----------
    matter_statistics : MatterStatistics
        MatterStatistics object carrying the cosmology.
    overdensity_type : str, optional
        Overdensity type passed to :class:`HaloProfileCore`.  Default ``"vir"``.
    overdensity : int, optional
        Overdensity value passed to :class:`HaloProfileCore`.  Default ``200``.
    two_halo : str, optional
        If ``"sum"``, the 1-halo and miscentered 2-halo profile are summed.
        If ``"max"``, the maximum between them is considered at each point.
        If ``"sum_exclusion"``, the 1-halo and the miscentered 2-halo profile with
        nonlinear power spectrum and halo exclusion inside R_Delta are summed
        (see ``HaloProfileCore.two_halo_exclusion``).
        If ``"None"``, the 2-halo is not included. Default is ``"None"``.
    trunc_fact : float, optional
        Truncation radius in units of the overdensity radius
        (:math:`R_t = \tau_\mathrm{vir} \cdot R_\Delta`).  Default ``3.0``.
    zs_max : float, optional
        Maximum source redshift for lensing calculations.
    exclusion_background : float or callable, optional
        Fraction of the mean density excluded inside R_Delta by the 2-halo term
        with exclusion (number or function of z). Default 1.
    R_min : float, optional
        Minimum radius of the integration grid (units : Mpc / h); the centred
        profile is taken constant below it in the mean within R. Default 1e-3.
    dlnR : float, optional
        Logarithmic spacing of the integration grid. Default 0.02.
    """

    def __init__(
        self,
        matter_statistics: MatterStatistics,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        two_halo: str = "None",
        trunc_fact: float = 3.0,
        zs_max: float = 2.0,
        exclusion_background=1.0,
        R_min: float = 1.0e-3,
        dlnR: float = 0.02,
    ):
        self.core = HaloProfileCore(
            matter_statistics,
            overdensity_type=overdensity_type,
            overdensity=overdensity,
            zs_max=zs_max,
            exclusion_background=exclusion_background,
        )
        # centred 1-halo profile
        self._centred = BMOHaloProfile(
            matter_statistics,
            overdensity_type=overdensity_type,
            overdensity=overdensity,
            two_halo="None",
            trunc_fact=trunc_fact,
            zs_max=zs_max,
        )
        self.two_halo = two_halo
        self.trunc_fact = trunc_fact
        self.R_min = R_min
        self.dlnR = dlnR
        # miscentered 1-halo profiles: the surface and excess surface densities are
        # computed together, and cached for the request of the other one
        self._cache_1h = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _operator(self, R, sigma_off):
        r"""
        Linear operator from the centred profile on the radial grid to the
        miscentered surface and excess surface densities at R.

        Parameters
        ----------
        R : np.ndarray
            Projected radii (Mpc/h), shape ``(R.size,)``.
        sigma_off : float
            Scale of the Rayleigh miscentering distribution (Mpc/h).

        Returns
        -------
        R_grid : np.ndarray
            Logarithmic radial grid (Mpc/h), shape ``(N,)``.
        operator : np.ndarray
            Shape ``(N, 2 R.size)``: ``Sigma(R_grid) @ operator`` gives
            ``[Sigma_off(R), DeltaSigma_off(R)]``.
        """
        R = np.ascontiguousarray(R, dtype=float)
        key = (float(sigma_off), R.tobytes(), self.R_min, self.dlnR)
        if key in _MISCENTERING_OPERATORS:
            return _MISCENTERING_OPERATORS[key]

        R_max = R.max() + 8.0 * sigma_off
        R_grid = np.exp(np.arange(np.log(self.R_min), np.log(R_max) + self.dlnR, self.dlnR))
        lnR_grid = np.log(R_grid)
        # Rice kernel for all the radii of the grid: (N, N), unit integral
        _, kernel = HaloProfileCore._rice_kernel(R_grid, sigma_off, np.inf, self.dlnR)
        # applied to unit profiles (rows): Sigma_off and DeltaSigma_off on the grid
        Sigma_off = kernel.T
        cumulative = (
            cumulative_trapezoid(Sigma_off * R_grid**2.0, lnR_grid, axis=-1, initial=0.0)
            + 0.5 * Sigma_off[:, :1] * self.R_min**2.0
        )
        DeltaSigma_off = 2.0 * cumulative / R_grid**2.0 - Sigma_off
        # linear interpolation in ln R at the requested radii
        lnR = np.log(R)
        j = np.clip(np.searchsorted(lnR_grid, lnR) - 1, 0, R_grid.size - 2)
        t = (lnR - lnR_grid[j]) / (lnR_grid[j + 1] - lnR_grid[j])
        operator = np.concatenate(
            [
                P[:, j] * (1.0 - t) + P[:, j + 1] * t
                for P in (Sigma_off, DeltaSigma_off)
            ],
            axis=1,
        )
        if len(_MISCENTERING_OPERATORS) > 256:
            _MISCENTERING_OPERATORS.clear()
        _MISCENTERING_OPERATORS[key] = (R_grid, operator)
        return R_grid, operator

    def _miscentered_1h(self, R_mpc, z, M, c, sigma_off):
        r"""
        Miscentered 1-halo surface and excess surface densities (h Msun / pc²),
        each with shape ``(z.size, M.size, R.size)``.

        ``R_mpc`` has shape ``(1, 1, R.size)`` or ``(z.size, 1, R.size)`` (radii
        depending on z, for angular units).
        """
        z = np.atleast_1d(z)
        key = tuple(
            np.ascontiguousarray(a, dtype=float).tobytes() for a in (R_mpc, z, M, c)
        ) + (float(sigma_off), np.shape(c))
        if key not in self._cache_1h:
            if len(self._cache_1h) > 512:
                self._cache_1h.clear()
            self._cache_1h[key] = self._compute_miscentered_1h(R_mpc, z, M, c, sigma_off)
        return self._cache_1h[key]

    def _compute_miscentered_1h(self, R_mpc, z, M, c, sigma_off):
        """See `_miscentered_1h`."""
        R_rows = R_mpc[:, 0, :]
        if sigma_off <= 0.0:
            # no miscentering
            out = []
            for i in range(z.size):
                c_i = c[i : i + 1] if np.ndim(c) == 2 else c
                R_i = R_rows[min(i, R_rows.shape[0] - 1)]
                out.append(
                    (
                        self._centred.surface_mass_density(R_i, z[i : i + 1], M, c_i),
                        self._centred.excess_surface_mass_density(R_i, z[i : i + 1], M, c_i),
                    )
                )
            return tuple(np.concatenate(p, axis=0) for p in zip(*out))
        if R_rows.shape[0] == 1:
            R_grid, operator = self._operator(R_rows[0], sigma_off)
            out = self._centred.surface_mass_density(R_grid, z, M, c) @ operator
        else:
            rows = []
            for i in range(z.size):
                R_grid, operator = self._operator(R_rows[i], sigma_off)
                c_i = c[i : i + 1] if np.ndim(c) == 2 else c
                rows.append(self._centred.surface_mass_density(R_grid, z[i : i + 1], M, c_i) @ operator)
            out = np.concatenate(rows, axis=0)
        n_R = R_rows.shape[1]
        return out[..., :n_R], out[..., n_R:]

    # ------------------------------------------------------------------
    # Public interface (as MiscBMOHaloProfileEmu)
    # ------------------------------------------------------------------

    def surface_mass_density(
        self,
        R: np.ndarray,
        z: np.ndarray,
        M: np.ndarray,
        c: float,
        sigma_off: float,
        halo_bias: np.ndarray = None,
        radius_units: str = "Mpc/h",
    ) -> np.ndarray:
        r"""
        Miscentered surface mass density profile :math:`\Sigma_\mathrm{off}`.

        Parameters
        ----------
        R : np.ndarray
            Projected radii (units controlled by ``radius_units``).
        z : np.ndarray
            Lens redshifts, shape ``(Nz,)``.
        M : np.ndarray
            Halo masses (Msun / h), shape ``(NM,)``.
        c : float or np.ndarray
            Concentration, scalar or with shape ``(Nz, NM)``.
        sigma_off : float
            Scatter of the Rayleigh miscentering PDF (Mpc/h),
            see Eq. 8 of `Johnston et al. 2007
            <https://arxiv.org/pdf/0709.1159.pdf>`_.
            It is also used for the miscentered 2-halo term.
        halo_bias : np.ndarray, optional
            Halo bias used for the 2h term, with shape ``(Nz, NM)``.
        radius_units : str, optional
            Unit for the input radii.  Accepted values are
            ``"Mpc/h"``, ``"radians"``, ``"degrees"``, ``"arcmin"``,
            ``"arcsec"``.  Default ``"Mpc/h"``.

        Returns
        -------
        Sigma_off : np.ndarray
            Miscentered surface mass density (h Msun / pc²),
            shape ``(Nz, NM, NR)``.
        """
        R_mpc, RDelta, _ = self.core.surface_mass_density_args(
            R, z, M, radius_units=radius_units
        )
        Sigma_off = self._miscentered_1h(R_mpc, z, M, c, sigma_off)[0]

        if self.two_halo != "None":
            Sigma_off = self.core.include_surface_mass_density_2h(
                Sigma_off,
                self.two_halo,
                R,
                z,
                halo_bias,
                radius_units,
                sigma_off=sigma_off,
                R_exc=RDelta[..., 0],
            )

        self.core.check_profile_shape(R, z, M, Sigma_off)

        return Sigma_off

    def excess_surface_mass_density(
        self,
        R: np.ndarray,
        z: np.ndarray,
        M: np.ndarray,
        c: float,
        sigma_off: float,
        halo_bias: np.ndarray = None,
        radius_units: str = "Mpc/h",
    ) -> np.ndarray:
        r"""
        Miscentered excess surface mass density profile
        :math:`\Delta\Sigma_\mathrm{off}`.

        Same parameters as `surface_mass_density`.

        Returns
        -------
        DeltaSigma_off : np.ndarray
            Miscentered excess surface mass density (h Msun / pc²),
            shape ``(Nz, NM, NR)``.
        """
        R_mpc, RDelta, _ = self.core.surface_mass_density_args(
            R, z, M, radius_units=radius_units
        )
        DeltaSigma_off = self._miscentered_1h(R_mpc, z, M, c, sigma_off)[1]

        if self.two_halo != "None":
            DeltaSigma_off = self.core.include_excess_surface_mass_density_2h(
                DeltaSigma_off,
                self.two_halo,
                R,
                z,
                halo_bias,
                radius_units,
                sigma_off=sigma_off,
                R_exc=RDelta[..., 0],
            )

        self.core.check_profile_shape(R, z, M, DeltaSigma_off)

        return DeltaSigma_off
