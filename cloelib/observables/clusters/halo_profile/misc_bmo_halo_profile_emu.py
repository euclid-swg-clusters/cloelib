import numpy as np

from cloelib.observables.clusters.matter_statistics import MatterStatistics

from .emu_net_weights import (
    delta_sigma_max_params as _DEFAULT_DSIGMA_MAX,
    delta_sigma_min_params as _DEFAULT_DSIGMA_MIN,
    delta_sigma_weights as _DEFAULT_DSIGMA_WEIGHTS,
    sigma_max_params as _DEFAULT_SIGMA_MAX,
    sigma_min_params as _DEFAULT_SIGMA_MIN,
    sigma_weights as _DEFAULT_SIGMA_WEIGHTS,
)
from .halo_profile_core import HaloProfileCore
from cloelib.cosmology import derived_cosmology


# ---------------------------------------------------------------------------
# Internal helper: lightweight NumPy feed-forward network
# ---------------------------------------------------------------------------


class _EmuNet:
    r"""
    Six-hidden-layer feed-forward network evaluated in NumPy.

    Architecture (halving hidden size at each layer)::

        input  →  fc1 (H)  →  fc2 (H/2)  →  fc3 (H/4)
               →  fc4 (H/8) →  fc5 (H/16) →  fc6 (output)

    Activation: Leaky ReLU (negative slope 0.01) after every hidden layer.
    The forward pass is evaluated in single precision (3-4x faster than double
    precision; differences of the output ~1e-5, well below the emulator accuracy).
    """

    def __init__(self, input_size: int, hidden_size: int = 512, output_size: int = 1):
        # Weight matrices  shape: (out_features, in_features)
        # Bias vectors     shape: (out_features,)
        sizes = [
            (hidden_size, input_size),
            (hidden_size // 2, hidden_size),
            (hidden_size // 4, hidden_size // 2),
            (hidden_size // 8, hidden_size // 4),
            (hidden_size // 16, hidden_size // 8),
            (output_size, hidden_size // 16),
        ]
        self._weights = [np.zeros(s) for s in sizes]
        self._biases = [np.zeros(s[0]) for s in sizes]
        self._set_single_precision()

    # ------------------------------------------------------------------
    def load_from_dict(self, weight_dict: dict) -> None:
        r"""
        Load pre-trained weights from a dictionary.

        The dictionary must contain keys ``fc1_w``, ``fc1_b``,
        ``fc2_w``, ``fc2_b``, …, ``fc6_w``, ``fc6_b`` mapping to
        numpy arrays.  This is the primary loading path when weights
        are embedded in :mod:`EmuNetWeights`.

        Parameters
        ----------
        weight_dict : dict
            Dictionary of weight and bias arrays.
        """
        for i in range(1, 7):
            self._weights[i - 1] = weight_dict[f"fc{i}_w"]
            self._biases[i - 1] = weight_dict[f"fc{i}_b"]
        self._set_single_precision()

    def _set_single_precision(self) -> None:
        """Transposed single-precision copies of the weights, used by `forward`."""
        self._weights_t32 = [np.ascontiguousarray(W.T, dtype=np.float32) for W in self._weights]
        self._biases32 = [np.asarray(b, dtype=np.float32) for b in self._biases]

    # ------------------------------------------------------------------
    def load_weights(self, npz_path: str) -> None:
        r"""
        Load pre-trained weights from a ``.npz`` file (convenience wrapper).

        Prefer :meth:`load_from_dict` with the weights from
        :mod:`EmuNetWeights` for path-independent deployment.

        Parameters
        ----------
        npz_path : str
            Path to the ``.npz`` weight file.
        """
        self.load_from_dict(dict(np.load(npz_path)))

    # ------------------------------------------------------------------
    @staticmethod
    def _leaky_relu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, x, 0.01 * x)

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        r"""
        Forward pass.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape ``(N, input_size)``.

        Returns
        -------
        out : np.ndarray
            Output array of shape ``(N, output_size)``.
        """
        out = np.asarray(x, dtype=np.float32)
        for i, (W_t, b) in enumerate(zip(self._weights_t32, self._biases32)):
            out = out @ W_t
            out += b
            if i < len(self._weights) - 1:  # no activation on output layer
                np.maximum(out, 0.01 * out, out=out)  # leaky ReLU
        return out.astype(np.float64)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MiscBMOHaloProfileEmu:
    r"""
    Miscentered BMO halo profile evaluated via neural-network emulators.

    Two independent six-hidden-layer networks (one for :math:`\Sigma_\mathrm{off}`,
    one for :math:`\Delta\Sigma_\mathrm{off}`) replace the numerical integration
    of the azimuthally averaged miscentered profiles. Each emulator predicts

    .. math::

        \ln\!\left(\frac{\Sigma_\mathrm{off}(R)}{\rho_s}\right)
        \quad\text{or}\quad
        \ln\!\left(\frac{\Delta\Sigma_\mathrm{off}(R)}{\rho_s}\right)

    as a function of the four input features
    :math:`[\log_{10} R,\, \log_{10} R_\mathrm{vir},\, c,\, \sigma_\mathrm{off}]`.

    The characteristic density :math:`\rho_s` is the virial BMO amplitude
    computed analytically from mass, concentration, and redshift (see
    :meth:`_rho_s_bmo`).

    The emulater has been trained setting tau_vir = 3.0 within these boundaries
    # Bounds #log10R [cMpc/h] # log10 Rvir [pMpc/h] # c # sigma_off
    lower_bounds = [-4., np.log10(0.15), 0.5, 0.05]
    upper_bounds = [np.log10(30.), np.log10(2.2), 10., 0.8]

    Parameters
    ----------
    matter_statistics : MatterStatistics
        MatterStatistics object carrying the cosmology.
    sigma_weights : dict, optional
        Weight dictionary for the :math:`\Sigma_\mathrm{off}` emulator,
        with keys ``fc1_w`` … ``fc6_w``, ``fc1_b`` … ``fc6_b``.
        Defaults to the weights embedded in :mod:`EmuNetWeights`.
    delta_sigma_weights : dict, optional
        Weight dictionary for the :math:`\Delta\Sigma_\mathrm{off}`
        emulator.  Defaults to the weights embedded in :mod:`EmuNetWeights`.
    sigma_min_params : array_like, shape (4,)
        Feature minima used to normalise the inputs of the
        :math:`\Sigma_\mathrm{off}` emulator.
        Defaults to the values in :mod:`EmuNetWeights`.
    sigma_max_params : array_like, shape (4,)
        Feature maxima used to normalise the inputs of the
        :math:`\Sigma_\mathrm{off}` emulator.
        Defaults to the values in :mod:`EmuNetWeights`.
    delta_sigma_min_params : array_like, shape (4,)
        Feature minima for the :math:`\Delta\Sigma_\mathrm{off}` emulator.
        Defaults to the values in :mod:`EmuNetWeights`.
    delta_sigma_max_params : array_like, shape (4,)
        Feature maxima for the :math:`\Delta\Sigma_\mathrm{off}` emulator.
        Defaults to the values in :mod:`EmuNetWeights`.
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
    hidden_size : int, optional
        Hidden-layer width of the emulator networks. Default ``512``.
    zs_max : float, optional
        Maximum source redshift for lensing calculations.
    exclusion_background : float or callable, optional
        Fraction of the mean density excluded inside R_Delta by the 2-halo term
        with exclusion (number or function of z). Default 1.

    Notes
    -----
    The emulators were trained following Eq. 8 of
    `Johnston et al. 2007 <https://arxiv.org/pdf/0709.1159.pdf>`_ for the
    miscentering PDF and the BMO truncated NFW profile of
    `Baltz et al. 2009 <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.
    """

    def __init__(
        self,
        matter_statistics: MatterStatistics,
        sigma_weights: dict = None,
        delta_sigma_weights: dict = None,
        sigma_min_params: np.ndarray = None,
        sigma_max_params: np.ndarray = None,
        delta_sigma_min_params: np.ndarray = None,
        delta_sigma_max_params: np.ndarray = None,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        two_halo: str = "None",
        trunc_fact: float = 3.0,
        hidden_size: int = 512,
        zs_max: float = 2.0,
        exclusion_background=1.0,
    ):
        self.core = HaloProfileCore(
            matter_statistics,
            overdensity_type=overdensity_type,
            overdensity=overdensity,
            zs_max=zs_max,
            exclusion_background=exclusion_background,
        )

        self.two_halo = two_halo
        self.trunc_fact = trunc_fact

        # Normalisation bounds — fall back to EmuNetWeights defaults
        self._sigma_min = np.asarray(
            sigma_min_params if sigma_min_params is not None else _DEFAULT_SIGMA_MIN
        )
        self._sigma_max = np.asarray(
            sigma_max_params if sigma_max_params is not None else _DEFAULT_SIGMA_MAX
        )
        self._dsigma_min = np.asarray(
            delta_sigma_min_params
            if delta_sigma_min_params is not None
            else _DEFAULT_DSIGMA_MIN
        )
        self._dsigma_max = np.asarray(
            delta_sigma_max_params
            if delta_sigma_max_params is not None
            else _DEFAULT_DSIGMA_MAX
        )

        # Instantiate and load the two emulator networks from weight dicts,
        # falling back to the weights embedded in EmuNetWeights
        self._emu_sigma = _EmuNet(input_size=4, hidden_size=hidden_size, output_size=1)
        self._emu_sigma.load_from_dict(
            sigma_weights if sigma_weights is not None else _DEFAULT_SIGMA_WEIGHTS
        )

        self._emu_delta_sigma = _EmuNet(
            input_size=4, hidden_size=hidden_size, output_size=1
        )
        self._emu_delta_sigma.load_from_dict(
            delta_sigma_weights
            if delta_sigma_weights is not None
            else _DEFAULT_DSIGMA_WEIGHTS
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rho_s_bmo(self, M: np.ndarray, c: float, z: np.ndarray) -> np.ndarray:
        r"""
        BMO characteristic density :math:`\rho_s`.

        Reproduces the analytical :math:`\rho_s` used during emulator training:

        .. math::

            \rho_s = \frac{\Delta_\mathrm{vir}(z)\, c^3}{3\, m_\mathrm{BMO}(c,\tau)}
            \cdot \rho_c(z)

        where :math:`\tau = \tau_\mathrm{vir} \cdot c` and
        :math:`m_\mathrm{BMO}` is the dimensionless BMO mass function
        (Eq. 2 of Johnston et al. 2007 / Eq. A of Baltz et al. 2009).

        Parameters
        ----------
        M : np.ndarray
            Halo mass (Msun / h), shape ``(M.size,)`` or scalar.
        c : float
            Concentration.
        z : np.ndarray
            Redshift, shape ``(z.size,)`` or scalar.

        Returns
        -------
        rho_s : np.ndarray
            Characteristic density (Msun h² / Mpc³),
            shape ``(z.size, M.size)``.
        """

        # Omega_m from cosmology
        Omega_m = self.core.matter_statistics.Omega_m_0
        h = self.core.matter_statistics.background.h

        z = np.atleast_1d(np.asarray(z, dtype=float))  # (Nz,)
        M = np.atleast_1d(np.asarray(M, dtype=float))  # (NM,)

        Ez2 = (
            self.core.matter_statistics.background.hubble_parameter(z) / (h * 100.0)
        ) ** 2.0
        rho_c_z = (
            derived_cosmology.rho_crit(self.core.matter_statistics.background, z)
            / h**2.0
        )  # (Nz,) Msun h^2/pMpc^3

        # Virial overdensity Delta_vir(z) w.r.t. critical density
        # Bryan & Norman (1998) fitting formula
        x = Omega_m * (1.0 + z) ** 3.0 / Ez2 - 1.0  # (Nz,)
        Delta_vir = 18.0 * np.pi**2.0 + 82.0 * x - 39.0 * x**2.0  # (Nz,)

        # Virial radius (Mpc/h), shape (Nz, NM)
        rho_c_z_2d = rho_c_z[:, np.newaxis]  # (Nz, 1)
        Delta_vir_2d = Delta_vir[:, np.newaxis]  # (Nz, 1)
        M_2d = M[np.newaxis, :]  # (1, NM)

        R_vir = (M_2d * 3.0 / (4.0 * np.pi * Delta_vir_2d * rho_c_z_2d)) ** (
            1.0 / 3.0
        )  # (Nz, NM)

        # BMO dimensionless mass function m_bmo(c, tau)
        tau = self.trunc_fact * c  # tau = R_t / R_s = trunc_fact * c

        term1 = (
            c
            * (tau**2.0 + 1.0)
            * (c * (c + 1.0) - tau**2.0 * (c - 1.0) * (2.0 + 3.0 * c) - 2.0 * tau**4.0)
        )
        term2 = (
            tau
            * (c + 1.0)
            * (tau**2.0 + c**2.0)
            * (
                2.0 * (3.0 * tau**2.0 - 1.0) * np.arctan(c / tau)
                + tau
                * (tau**2.0 - 3.0)
                * np.log(tau**2.0 * (1.0 + c) ** 2.0 / (tau**2.0 + c**2.0))
            )
        )
        m_bmo = (
            tau**2.0
            / (2.0 * (tau**2.0 + 1.0) ** 3.0 * (1.0 + c) * (tau**2.0 + c**2.0))
            * (term1 + term2)
        )

        # rho_s = Delta_vir * c³ / (3 * m_bmo) * rho_c(z),  shape (Nz, NM)
        rho_s = rho_c_z_2d * Delta_vir_2d * c**3.0 / (3.0 * m_bmo)

        return rho_s, R_vir  # both (Nz, NM)

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(X: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
        r"""Min-max normalisation to [0, 1]."""
        return (X - x_min) / (x_max - x_min)

    # ------------------------------------------------------------------

    def _predict(
        self,
        emu: _EmuNet,
        x_min: np.ndarray,
        x_max: np.ndarray,
        R: np.ndarray,
        R_vir: np.ndarray,
        c: float,
        sigma_off: float,
        rho_s: np.ndarray,
    ) -> np.ndarray:
        r"""
        Run one emulator over the full (z, M, R) grid.

        Parameters
        ----------
        emu : _EmuNet
            The emulator network to evaluate.
        x_min, x_max : np.ndarray
            Normalisation bounds, shape (4,).
        R : np.ndarray
            Projected radii (Mpc/h), shape ``(R.size,)``, or ``(z.size, R.size)``
            for radii depending on z (angular units).
        R_vir : np.ndarray
            Virial radii (Mpc/h), shape ``(z.size, M.size)``.
        c : float
            Concentration.
        sigma_off : float
            Miscentering scatter (Mpc/h).
        rho_s : np.ndarray
            Characteristic density (Msun h²/Mpc³), shape ``(z.size, M.size)``.

        Returns
        -------
        profile : np.ndarray
            Predicted profile (h Msun/pc²), shape ``(z.size, M.size, R.size)``.
        """
        Nz, NM = R_vir.shape
        NR = np.shape(R)[-1]

        # Build feature array: (Nz * NM * NR, 4)
        log10_R = np.log10(np.atleast_2d(R))  # (1 or Nz, NR)
        log10_Rvir = np.log10(R_vir)  # (Nz, NM)

        # Broadcast to (Nz, NM, NR)
        log10_R_grid = np.broadcast_to(
            log10_R[:, np.newaxis, :], (Nz, NM, NR)
        ).reshape(-1)
        log10_Rvir_grid = np.broadcast_to(
            log10_Rvir[:, :, np.newaxis], (Nz, NM, NR)
        ).reshape(-1)
        # c is a scalar or has shape (Nz, NM)
        c_grid = np.broadcast_to(
            np.asarray(c, dtype=float)[..., np.newaxis] if np.ndim(c) == 2 else c,
            (Nz, NM, NR),
        ).reshape(-1)
        sigma_off_grid = np.full(Nz * NM * NR, sigma_off)

        X = np.column_stack(
            [log10_R_grid, log10_Rvir_grid, c_grid, sigma_off_grid]
        )  # (Nz*NM*NR, 4)

        X_norm = self._normalise(X, x_min, x_max)
        Y_pred = emu.forward(X_norm).reshape(-1)  # (Nz*NM*NR,)

        # Undo log-scaling and multiply by rho_s
        rho_s_grid = np.broadcast_to(rho_s[:, :, np.newaxis], (Nz, NM, NR)).reshape(-1)

        profile_flat = np.exp(Y_pred) * rho_s_grid  # Msun h² / Mpc³ · Mpc
        # Convert to h Msun / pc²: 1 Mpc = 1e6 pc  →  1/Mpc² = 1e-12 /pc²
        profile_flat *= 1.0e-12

        return profile_flat.reshape(Nz, NM, NR)

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

        rho_s, R_vir = self._rho_s_bmo(M, c, z)

        Sigma_off = self._predict(
            self._emu_sigma,
            self._sigma_min,
            self._sigma_max,
            R_mpc[:, 0, :],  # (1, R.size), or (z.size, R.size) for angular units
            R_vir,
            c,
            sigma_off,
            rho_s,
        )

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

    # ------------------------------------------------------------------

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
        DeltaSigma_off : np.ndarray
            Miscentered excess surface mass density (h Msun / pc²),
            shape ``(Nz, NM, NR)``.
        """
        R_mpc, RDelta, _ = self.core.surface_mass_density_args(
            R, z, M, radius_units=radius_units
        )

        rho_s, R_vir = self._rho_s_bmo(M, c, z)

        DeltaSigma_off = self._predict(
            self._emu_delta_sigma,
            self._dsigma_min,
            self._dsigma_max,
            R_mpc[:, 0, :],
            R_vir,
            c,
            sigma_off,
            rho_s,
        )

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
