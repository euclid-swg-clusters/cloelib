import numpy as np
from scipy import interpolate
from scipy.integrate import quad_vec
from scipy.special import j0, j1

from cloelib.auxiliary import units
from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations

from .auxiliary import convert_distance


def _bessel_j2(x):
    """Bessel function j2"""
    return 2.0 / x * j1(x) - j0(x)


# kernels of the tabulated projected 2-halo integrals and nonlinear correlation function, shared
# by all the MatterStatistics instances with the same k range (they do not depend on the cosmology)
_PROJECTED_2H_KERNELS = {}


class MatterStatistics:
    def __init__(
        self,
        perturbations: Perturbations,
        interpolate_pk: bool = True,
        interpolate_da: bool = True,
        z=np.linspace(1.0e-5, 2.0 - 1.0e-5, 100),
        k=np.geomspace(1e-4, 10, 500),
        nonlinear_perturbations: Perturbations = None,
        tabulate_2h: bool = True,
    ):
        r"""Auxiliary class computing quantities used in galaxy cluster models.

        Initialize the class with given perturbations and overdensity definition.
        If requested, cosmological functions (e.g., the matter power spectrum)
        are interpolated.

        Parameters
        ----------
        perturbations : Perturbations
            An object from the `LinearPerturbations` class containing cosmological
            perturbation data (e.g., power spectrum, growth function).
        interpolate_pk : bool, optional
            If true, the class interpolates the matter power spectrum.
            A default interpolation is set when class is instanciated with
            interpolate_pk=True. For a more customized interpolation, check
            the set_matter_power_spectrum_interpolation function.
        interpolate_da : bool, optional
            If true, the class interpolates the angular diameter distance.
            A default interpolation is set when class is instanciated with
            interpolate_da=True. For a more customized interpolation, check
            the set_angular_diameter_distance_interpolation function.
        nonlinear_perturbations : Perturbations, optional
            Object providing the nonlinear cold dark matter + baryons power
            spectrum (`matter_power_spectrum_cb`), used by the 2-halo term with
            halo exclusion (`two_halo="sum_exclusion"` of the halo profiles).
            It must share the background of `perturbations`.
        tabulate_2h : bool, optional
            If True (default), the k integral of the projected 2-halo profiles is
            computed once per redshift on a fixed logarithmic k grid and tabulated in
            comoving radius (see `_projected_2h_integral`), instead of an adaptive
            integration at each call.

        Notes
        ----------
        In the current implementation, the matter power spectrum never includes
        the contribution of massive neutrinos. Halo mass function, halo bias, and
        2-halo profile models implemented in this subpackage require
        cold dark matter + baryons (no neutrinos) power spectra.
        """
        self.perturbations = perturbations
        self.nonlinear_perturbations = nonlinear_perturbations
        self.k = k

        # Interpolators
        self.Pk_interp_cb = None
        self.Pk_interp_cb_nl = None
        self.da_interp = None
        # tables of the nonlinear correlation function, for each redshift
        self._xi_nl_tables = {}
        # tables of the projected 2-halo integrals, for each (Bessel function, P(k), redshift)
        self.tabulate_2h = tabulate_2h
        self._projected_2h_tables = {}

        # set P(k) interpolation usage
        if interpolate_pk:
            self.set_matter_power_spectrum_interpolation(z, self.k)
        self.interpolate_pk = interpolate_pk

        # set angular diameter distance interpolation usage
        if interpolate_da:
            self.set_angular_diameter_distance_interpolation(z)
        self.interpolate_da = interpolate_da

        # Density parameters at z=0
        self.Omega_m_0 = self.background.Omega_m(0.0)
        self.Omega_cb_0 = self.background.Omega_cb(0.0)
        self.Omega_b_0 = self.background.Omega_b(0.0)

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.perturbations.background

    @property
    def interpolate_pk(self):
        r"""If true, class uses interpolation for matter power spectrum computation."""
        return self.__interpolate_pk

    @property
    def interpolate_da(self):
        r"""If true, class uses interpolation for angular diameter distance computation."""
        return self.__interpolate_da

    @interpolate_pk.setter
    def interpolate_pk(self, interpolate_pk):
        """If true, makes class uses interpolation for matter power spectrum computation."""
        if interpolate_pk:
            self.matter_power_spectrum_cb = self.Pk_interp_cb
            self.matter_power_spectrum_cb_nl = self.Pk_interp_cb_nl
        else:
            self.matter_power_spectrum_cb = _matter_power_spectrum_cb_exact
            self.matter_power_spectrum_cb_nl = self._matter_power_spectrum_cb_nl_exact
        self.__interpolate_pk = interpolate_pk

    @interpolate_da.setter
    def interpolate_da(self, interpolate_da):
        """If true, makes class uses interpolation for angular diameter distance computation."""
        if interpolate_da:
            self.angular_diameter_distance = self.da_interp
        else:
            self.angular_diameter_distance = self.background.angular_diameter_distance
        self.__interpolate_da = interpolate_da

    def _matter_power_spectrum_cb_exact(self, z, k):
        r"""Computes the non interpolated matter power spectrum.

        This function computes the cold dark matter + baryons power spectrum,
        not including the massive neutrino contribution.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        k: float or np.ndarray
            Wavenumber where W(kR) is evaluated.
            Units: h Mpc^{-1}

        Returns
        -------
        float or np.ndarray
            Matter power spectrum.
        """
        return self.perturbations.matter_power_spectrum_cb(
            z,
            k,
            hubble_units=True,
            k_hunit=True,
        )

    def _matter_power_spectrum_cb_nl_exact(self, z, k):
        r"""Computes the non interpolated nonlinear matter power spectrum.

        This function computes the nonlinear cold dark matter + baryons power
        spectrum, not including the massive neutrino contribution.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        k: float or np.ndarray
            Wavenumber.
            Units: h Mpc^{-1}

        Returns
        -------
        float or np.ndarray
            Nonlinear matter power spectrum.
        """
        if self.nonlinear_perturbations is None:
            raise ValueError(
                "The nonlinear power spectrum requires nonlinear_perturbations in MatterStatistics."
            )
        return self.nonlinear_perturbations.matter_power_spectrum_cb(
            z,
            k,
            hubble_units=True,
            k_hunit=True,
        )

    def set_matter_power_spectrum_interpolation(self, z, k):
        r"""Create internal interpolation of matter power spectrum.

        This function interpolates the cold dark matter + baryons power spectrum,
        not including the massive neutrino contribution.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        k: float or np.ndarray
            Wavenumber where W(kR) is evaluated.
            Units: h Mpc^{-1}
        """
        # Power spectrum interpolation
        self.Pk_interp_cb = interpolate.RectBivariateSpline(
            z,
            k,
            self._matter_power_spectrum_cb_exact(z, k),
        )
        if self.nonlinear_perturbations is not None:
            self.Pk_interp_cb_nl = interpolate.RectBivariateSpline(
                z,
                k,
                self._matter_power_spectrum_cb_nl_exact(z, k),
            )

    def set_angular_diameter_distance_interpolation(self, z):
        r"""Create internal interpolation of angular diameter distance.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        """
        self.da_interp = interpolate.InterpolatedUnivariateSpline(
            x=z,
            y=self.background.angular_diameter_distance(z),
            ext=2,
        )

    def _generic_mass_density_2h(
        self, R, z, bessel_function, radius_units="Mpc/h", nonlinear=False
    ):
        r"""
        Surface or excess surface 2-halo density profile.

        Computes either the cosmological unbiased surface or excess surface
        (depending on the input Bessel function) 2-halo density profile.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        bessel_function: function
            Bessel function that goes in the integrand with the power spectrum.
            Used to return the surface density or the excess surface density.
            It should take (ll*theta) as input.
        nonlinear: bool, optional
            If True, use the nonlinear power spectrum.

        Returns
        -------
        profile: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, R.size).
        """
        power_spectrum = (
            self.matter_power_spectrum_cb_nl if nonlinear else self.matter_power_spectrum_cb
        )
        if power_spectrum is None:
            raise ValueError(
                "The nonlinear power spectrum requires nonlinear_perturbations in MatterStatistics."
            )

        # Calculate all redshift base quantities and shape them (nz, 1)
        z_plus_1 = np.asarray(z)[:, np.newaxis] + 1.0
        D_A = self.angular_diameter_distance(z)[:, np.newaxis] * self.background.h
        rho_m = (
            self.background.Omega_m(z)
            * derived_cosmology.rho_crit(self.background, z)
            / self.background.h**2
        )[:, np.newaxis]

        ## 1. Get radial distance in radians
        theta = convert_distance(R, radius_units, "radians", D_A)
        if radius_units.lower() != "mpc/h":
            # in this case, theta was missing z dimension
            theta = theta[np.newaxis, :]

        if self.tabulate_2h:
            # comoving radius, k (1 + z) D_A theta = k R_com
            R_com = theta * z_plus_1 * D_A
            integral = self._projected_2h_integral(R_com, z, bessel_function, nonlinear)
            if integral is not None:
                return 1.0e-12 * rho_m * integral / (2.0 * np.pi * z_plus_1)

        ## 2. Integrand function
        def integrand(kl):
            ll = kl * z_plus_1 * D_A
            Pk_vals = power_spectrum(z, kl)
            return bessel_function(ll * theta) * ll * Pk_vals

        ## 3. Integration
        two_point_corr = (
            quad_vec(integrand, self.k.min(), self.k.max(), epsrel=1e-1)[0]
            * z_plus_1
            * D_A
        )

        # Final strictly 2D calculation
        profile = (1.0e-12 * rho_m * two_point_corr) / (
            2.0 * np.pi * z_plus_1**3.0 * D_A**2.0
        )

        return profile

    def _projected_2h_integral(
        self, R_com, z, bessel_function, nonlinear, k_switch=1.0, n_k_log=1000, dk=0.01,
        r_min=1.0e-4, r_max=100.0, n_r=500,
    ):
        r"""
        Tabulated k integral of the projected 2-halo profiles.

        .. math::

            I(R, z) = \int_{k_{\min}}^{k_{\max}} dk\, k\, J(k R)\, P(k, z)

        with :math:`J = J_0` (surface density) or :math:`J_2` (excess surface density),
        over the range of the power spectrum tables (`self.k`), computed with the
        trapezoidal rule and tabulated in :math:`\ln R` (cubic spline) once for each
        redshift. The k grid is logarithmic below `k_switch` and linear above it, with
        step `dk`, which samples the oscillations of the Bessel function at all k (the
        integrand of the nonlinear power spectrum grows up to the maximum k).
        The sharp cut at :math:`k_{\max}` gives a term oscillating in R with period
        :math:`2\pi/k_{\max}`, :math:`\pm k_{\max} J_1(k_{\max} R) P(k_{\max}) / R` (+ for
        :math:`J_0`, - for :math:`J_2`, from the integration by parts), which is added
        analytically to the tabulated (smooth) remainder.
        Relative accuracy ~1e-4 up to R = 30 Mpc/h (~5e-4 at 50 Mpc/h, ~1e-3 at 100 Mpc/h).

        Parameters
        ----------
        R_com: np.ndarray
            Comoving radius (units : Mpc / h), shape (z.size, R.size).
        z: np.ndarray
            Redshift.
        bessel_function: function
            j0 or _bessel_j2.
        nonlinear: bool
            If True, use the nonlinear power spectrum.
        k_switch: float, optional
            Wavenumber of the switch from logarithmic to linear k grid (units : h / Mpc).
        n_k_log: int, optional
            Number of points of the logarithmic k grid.
        dk: float, optional
            Step of the linear k grid (units : h / Mpc).
        r_min, r_max: float, optional
            Range of the table in comoving radius (units : Mpc / h). Outside it, the
            adaptive integration is used.
        n_r: int, optional
            Number of points of the table.

        Returns
        -------
        integral: np.ndarray or None
            Shape (z.size, R.size), or None if some radius is outside the table.
        """
        z = np.atleast_1d(z)
        R_com = np.broadcast_to(R_com, (z.size, np.shape(R_com)[-1]))
        if R_com.min() < r_min or R_com.max() > r_max:
            return None
        k_min, k_max = float(self.k.min()), float(self.k.max())
        key_kernel = (bessel_function.__name__, k_min, k_max, k_switch, n_k_log, dk, r_min, r_max, n_r)
        if key_kernel not in _PROJECTED_2H_KERNELS:
            k_switch = min(max(k_switch, k_min), k_max)
            k = np.concatenate(
                [np.geomspace(k_min, k_switch, n_k_log), np.arange(k_switch, k_max, dk)[1:], [k_max]]
            )
            weights = np.zeros(k.size)
            weights[1:] += 0.5 * np.diff(k)
            weights[:-1] += 0.5 * np.diff(k)
            r_tab = np.geomspace(r_min, r_max, n_r)
            # J(kR) k dk : (r, k)
            kernel = bessel_function(r_tab[:, np.newaxis] * k[np.newaxis, :]) * (weights * k)[np.newaxis, :]
            _PROJECTED_2H_KERNELS[key_kernel] = (k, r_tab, kernel)
        k, r_tab, kernel = _PROJECTED_2H_KERNELS[key_kernel]
        # boundary term of the cut at k_max: sign * k_max J_1(k_max R) / R (times P(k_max))
        sign = {"j0": 1.0, "_bessel_j2": -1.0}.get(bessel_function.__name__, 0.0)

        def boundary(r):
            return sign * k_max * j1(k_max * r) / r

        power_spectrum = self.matter_power_spectrum_cb_nl if nonlinear else self.matter_power_spectrum_cb
        tables = self._projected_2h_tables
        missing = [zz for zz in np.unique(z) if (key_kernel, nonlinear, float(zz)) not in tables]
        if missing:
            pk = np.atleast_2d(power_spectrum(np.asarray(missing), k))
            for zz, pk_z in zip(missing, pk):
                tables[(key_kernel, nonlinear, float(zz))] = (
                    interpolate.CubicSpline(np.log(r_tab), kernel @ pk_z - pk_z[-1] * boundary(r_tab)),
                    pk_z[-1],
                )
        out = np.empty(R_com.shape)
        for i, zz in enumerate(z):
            smooth, pk_max = tables[(key_kernel, nonlinear, float(zz))]
            out[i] = smooth(np.log(R_com[i])) + pk_max * boundary(R_com[i])
        return out

    def surface_mass_density_2h(self, R, z, radius_units="Mpc/h", nonlinear=False):
        r"""
        Surface 2-halo matter density profile.

        Computes the cosmological unbiased surface 2-halo density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        nonlinear: bool, optional
            If True, use the nonlinear power spectrum.

        Returns
        -------
        Sigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, R.size).
        """
        return self._generic_mass_density_2h(
            R, z, bessel_function=j0, radius_units=radius_units, nonlinear=nonlinear
        )

    def excess_surface_mass_density_2h(self, R, z, radius_units="Mpc/h", nonlinear=False):
        r"""
        Excess surface 2-halo matter density profile.

        Computes the cosmological unbiased excess surface 2-halo
        density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        nonlinear: bool, optional
            If True, use the nonlinear power spectrum.

        Returns
        -------
        DeltaSigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, R.size).
        """
        return self._generic_mass_density_2h(
            R, z, bessel_function=_bessel_j2, radius_units=radius_units, nonlinear=nonlinear
        )

    def correlation_function_cb_nl(self, r, z, n_k=4096, r_min=1.0e-2, r_max=300.0, n_r=400):
        r"""
        Nonlinear cold dark matter + baryons correlation function.

        .. math::

            \xi_{\rm NL}(r, z) = \int \frac{k^3 P_{\rm NL}(k, z)}{2\pi^2}\,
            j_0(kr)\, d\ln k

        computed on a logarithmic k grid over the range of the power spectrum
        tables (`self.k`) and tabulated in :math:`\ln r` once for each redshift.
        The high-k end is smoothed with a Gaussian damping at the maximum k of
        the tables, which suppresses the ringing of the sharp cut. Accurate for
        :math:`r \gtrsim 10/k_{\max}`, as needed by the 2-halo term with halo
        exclusion (only :math:`r > R_\Delta` is used).

        Parameters
        ----------
        r: np.ndarray
            Comoving separation (units : Mpc / h).
        z: np.ndarray
            Redshift.
        n_k: int, optional
            Number of points of the k grid.
        r_min, r_max: float, optional
            Range of the table in r (units : Mpc / h); outside it the correlation
            function is extrapolated linearly in ln r.
        n_r: int, optional
            Number of points of the table in r.

        Returns
        -------
        xi: np.ndarray
            Correlation function, shape (z.size, r.size).
        """
        z = np.atleast_1d(z)
        r = np.atleast_1d(r)
        out = np.empty((z.size, r.size))
        missing = [zz for zz in np.unique(z) if float(zz) not in self._xi_nl_tables]
        if missing:
            # j0(kr) weighted by the integration measure: (r, k), shared by all the instances
            # with the same k range (it does not depend on the cosmology)
            key_kernel = ("xi", np.log(self.k.min()), np.log(self.k.max()), n_k, r_min, r_max, n_r)
            if key_kernel not in _PROJECTED_2H_KERNELS:
                lnk = np.linspace(np.log(self.k.min()), np.log(self.k.max()), n_k)
                k = np.exp(lnk)
                r_tab = np.geomspace(r_min, r_max, n_r)
                damping = np.exp(-((k / (0.5 * self.k.max())) ** 2))
                weights = np.full(n_k, lnk[1] - lnk[0])
                weights[[0, -1]] *= 0.5
                kr = r_tab[:, np.newaxis] * k[np.newaxis, :]
                kernel = np.sinc(kr / np.pi) * (weights * damping * k**3 / (2.0 * np.pi**2))[np.newaxis, :]
                _PROJECTED_2H_KERNELS[key_kernel] = (k, r_tab, kernel)
            k, r_tab, kernel = _PROJECTED_2H_KERNELS[key_kernel]
            pk = np.atleast_2d(self.matter_power_spectrum_cb_nl(np.asarray(missing), k))
            for zz, pk_z in zip(missing, pk):
                self._xi_nl_tables[float(zz)] = interpolate.InterpolatedUnivariateSpline(
                    np.log(r_tab), kernel @ pk_z, k=3
                )
        for i, zz in enumerate(z):
            out[i] = self._xi_nl_tables[float(zz)](np.log(r))
        return out
