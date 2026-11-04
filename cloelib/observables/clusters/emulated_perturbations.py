"""Linear and nonlinear power spectrum emulators with the interface of the cluster modules"""

import warnings

import numpy as np
from scipy import interpolate

from cloelib.cosmology.camb_cosmology import ensure_z_zero_included
from cloelib.cosmology.cosmology import Background


class EmulatorPerturbations:
    def __init__(self, background: Background, matter=None, matter_cb=None):
        r"""Power spectrum emulators with the interface used by the cluster modules.

        The cluster modules (`MatterStatistics`, `HaloCovariance`,
        `ClusterWeakLensingCovariance`) call `matter_power_spectrum_cb` and
        `matter_power_spectrum` with the `hubble_units` and `k_hunit` arguments of the
        CAMB perturbations. This class wraps emulators that return the power spectrum
        in Mpc^3 at wavenumbers in 1/Mpc through ``matter_power_spectrum(zs, ks)``, as
        the CosmoPower-JAX classes of `cloelib.cosmology.cosmopower_jax_cosmology`
        (e.g. ``CosmoPowerJAXLCDMPerturbations.LinearCB`` for the cold dark matter +
        baryons spectrum, ``.NonLinearCB`` for the nonlinear one), so that they can be
        passed as `perturbations` or `nonlinear_perturbations` of `MatterStatistics`.

        Parameters
        ----------
        background : Background
            Background of the emulated cosmology.
        matter : object, optional
            Emulator of the total matter power spectrum, ``matter_power_spectrum(zs, ks)``
            (units: Mpc^3, ks in 1/Mpc). Needed by `matter_power_spectrum` (e.g. the
            convergence power spectrum of the lensing covariance).
        matter_cb : object, optional
            Emulator of the cold dark matter + baryons power spectrum, same interface.
            Needed by `matter_power_spectrum_cb` (halo mass function, bias, 2-halo term).

        Examples
        --------
        >>> emu = CosmoPowerJAXLCDMPerturbations
        >>> linear = EmulatorPerturbations(
        ...     background, emu.Linear(background, z), emu.LinearCB(background, z)
        ... )
        >>> matter_stat = MatterStatistics(linear, z=z_tab, k=k_tab)
        """
        if matter is None and matter_cb is None:
            raise ValueError("EmulatorPerturbations requires matter and/or matter_cb.")
        self._background = background
        self.matter = matter
        self.matter_cb = matter_cb

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self._background

    def _power_spectrum(self, emulator, name, zs, ks, hubble_units, k_hunit):
        if emulator is None:
            raise ValueError(f"EmulatorPerturbations: no emulator for {name}.")
        h = self.background.h
        zs, ks = np.atleast_1d(zs), np.atleast_1d(ks)
        pk = np.asarray(emulator.matter_power_spectrum(zs, ks * h if k_hunit else ks))
        return pk * h**3 if hubble_units else pk

    def matter_power_spectrum(self, zs, ks, hubble_units=False, k_hunit=False):
        r"""Total matter power spectrum, shape (zs.size, ks.size).

        Parameters
        ----------
        zs: numpy.ndarray
            Redshifts (increasing).
        ks: numpy.ndarray
            Wavenumbers (increasing), in h/Mpc if `k_hunit`, else 1/Mpc.
        hubble_units: bool, optional
            If True, the output is in (Mpc/h)^3, else in Mpc^3.
        k_hunit: bool, optional
            If True, `ks` is in h/Mpc.
        """
        return self._power_spectrum(self.matter, "matter", zs, ks, hubble_units, k_hunit)

    def matter_power_spectrum_cb(self, zs, ks, hubble_units=False, k_hunit=False):
        r"""Cold dark matter + baryons power spectrum, shape (zs.size, ks.size).

        Same parameters as `matter_power_spectrum`.
        """
        return self._power_spectrum(self.matter_cb, "matter_cb", zs, ks, hubble_units, k_hunit)

    def growth_factor(self, zs, ks):
        r"""Growth factor of the wrapped emulator (cb if available)."""
        emulator = self.matter_cb if self.matter_cb is not None else self.matter
        return emulator.growth_factor(zs, ks)

    def sigma8_0(self):
        r"""sigma8 at z = 0 of the wrapped emulator (total matter if available)."""
        emulator = self.matter if self.matter is not None else self.matter_cb
        return emulator.sigma8_0()


class NonLinearBoostPerturbations:
    _boost_emulator = None

    def __init__(self, linear, redshifts, cmin=3.13, eta_0=0.603):
        r"""Nonlinear power spectra from the linear ones and the HMcode boost emulator.

        .. math::

            P_{\rm NL}(k, z) = P_{\rm L}(k, z)\, B(k, z)

        with the boost B = P_NL / P_L of the CosmoPower-JAX ``mpk_boost`` emulator
        (HMcode 2016, Mead et al. 2016), dark matter only for the default halo parameters
        (cmin = 3.13, eta_0 = 0.603). The boost is emulated for k <= ~10 1/Mpc and is
        extrapolated in log-log with the slope of its last points above it (up to
        100 h/Mpc), and taken constant below its minimum k. It can be passed as the
        `nonlinear_perturbations` of `MatterStatistics` (2-halo term with halo
        exclusion) or of `ClusterWeakLensingCovariance`.

        Parameters
        ----------
        linear : object
            Linear perturbations with the interface of the cluster modules (e.g.
            `CAMBLinearPerturbations`, `EmulatorPerturbations`):
            ``matter_power_spectrum[_cb](zs, ks, hubble_units, k_hunit)`` and `background`.
        redshifts : numpy.ndarray
            Redshifts of the boost table (z = 0 is added).
        cmin, eta_0 : float, optional
            HMcode 2016 halo concentration amplitude and halo bloating parameter.
        """
        if NonLinearBoostPerturbations._boost_emulator is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

                NonLinearBoostPerturbations._boost_emulator = CosmoPowerJAX(
                    probe="mpk_boost", verbose=False
                )
        emu = NonLinearBoostPerturbations._boost_emulator
        self.linear = linear
        bg = linear.background
        h = bg.h
        z = np.sort(ensure_z_zero_included(np.asarray(redshifts, dtype=float)))
        params = {
            "omega_b": bg.Omega_b0 * h**2,
            "omega_cdm": bg.Omega_cdm0 * h**2,
            "h": h,
            "n_s": bg.ns,
            "ln10^{10}A_s": np.log(bg.As * 1e10),
            "cmin": cmin,
            "eta_0": eta_0,
        }
        params = {key: np.full(z.size, value) for key, value in params.items()}
        params["z"] = z
        ln_b = np.log(np.array(emu.predict(params)))
        ln_k = np.log(np.asarray(emu.modes) / h)  # h/Mpc
        # log-log extrapolation above the maximum k of the emulator
        slope = (ln_b[:, -1] - ln_b[:, -20]) / (ln_k[-1] - ln_k[-20])
        ln_k_ext = np.log(np.geomspace(np.exp(ln_k[-1]) * 1.01, 100.0, 40))
        ln_b = np.hstack([ln_b, ln_b[:, -1:] + slope[:, np.newaxis] * (ln_k_ext[np.newaxis, :] - ln_k[-1])])
        self.ln_boost = interpolate.RectBivariateSpline(
            z, np.concatenate([ln_k, ln_k_ext]), ln_b, kx=1, ky=1
        )

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.linear.background

    def boost(self, zs, ks, k_hunit=False):
        r"""Boost P_NL / P_L, shape (zs.size, ks.size)."""
        ks = np.atleast_1d(ks) / (1.0 if k_hunit else self.background.h)  # h/Mpc
        return np.exp(self.ln_boost(np.atleast_1d(zs), np.log(ks)))

    def matter_power_spectrum(self, zs, ks, hubble_units=False, k_hunit=False):
        r"""Nonlinear total matter power spectrum, shape (zs.size, ks.size).

        Same parameters as `EmulatorPerturbations.matter_power_spectrum`.
        """
        return self.linear.matter_power_spectrum(
            zs, ks, hubble_units=hubble_units, k_hunit=k_hunit
        ) * self.boost(zs, ks, k_hunit)

    def matter_power_spectrum_cb(self, zs, ks, hubble_units=False, k_hunit=False):
        r"""Nonlinear cold dark matter + baryons power spectrum, shape (zs.size, ks.size).

        Same parameters as `EmulatorPerturbations.matter_power_spectrum`.
        """
        return self.linear.matter_power_spectrum_cb(
            zs, ks, hubble_units=hubble_units, k_hunit=k_hunit
        ) * self.boost(zs, ks, k_hunit)
