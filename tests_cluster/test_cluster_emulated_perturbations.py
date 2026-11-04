import numpy as np
import pytest
from numpy.testing import assert_allclose

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.emulated_perturbations import (
    EmulatorPerturbations,
    NonLinearBoostPerturbations,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics

background = CAMBBackground(
    H0=67.0,
    Omega_cdm0=0.12 / 0.67**2,
    Omega_b0=0.022 / 0.67**2,
    Omega_k0=0.0,
    w0=-1.0,
    wa=0.0,
    ns=0.96,
    mnu=0.06,
    As=2e-9,
    gamma_MG=0.0,
    N_mnu=1,
)
z_tab = np.linspace(0.0, 2.0, 41)
camb_linear = CAMBLinearPerturbations(background, z_tab)


class _MpcEmulator:
    """Emulator-like object (as the CosmoPower-JAX classes): P in Mpc^3, k in 1/Mpc"""

    def __init__(self, cb):
        self.function = camb_linear.matter_power_spectrum_cb if cb else camb_linear.matter_power_spectrum

    def matter_power_spectrum(self, zs, ks):
        return self.function(zs, ks)


def test_emulator_perturbations():
    print("# EmulatorPerturbations")
    emulated = EmulatorPerturbations(background, _MpcEmulator(cb=False), _MpcEmulator(cb=True))
    z, k = np.array([0.0, 0.5, 1.0]), np.geomspace(1e-3, 5.0, 30)
    for name in ("matter_power_spectrum", "matter_power_spectrum_cb"):
        for hubble_units in (False, True):
            for k_hunit in (False, True):
                assert_allclose(
                    getattr(emulated, name)(z, k, hubble_units=hubble_units, k_hunit=k_hunit),
                    getattr(camb_linear, name)(z, k, hubble_units=hubble_units, k_hunit=k_hunit),
                    rtol=1e-6,
                )
    # in MatterStatistics
    ms_kw = dict(z=np.linspace(1e-5, 2.0 - 1e-5, 50), k=np.geomspace(1e-4, 10.0, 300))
    R, zz = np.geomspace(0.3, 10.0, 5), np.array([0.3, 0.8])
    assert_allclose(
        MatterStatistics(emulated, **ms_kw).excess_surface_mass_density_2h(R, zz),
        MatterStatistics(camb_linear, **ms_kw).excess_surface_mass_density_2h(R, zz),
        rtol=1e-6,
    )
    with pytest.raises(ValueError):
        EmulatorPerturbations(background, matter=_MpcEmulator(cb=False)).matter_power_spectrum_cb(z, k)


def test_nonlinear_boost_perturbations():
    pytest.importorskip("cosmopower_jax")
    print("# NonLinearBoostPerturbations")
    nonlinear = NonLinearBoostPerturbations(camb_linear, z_tab)
    z, k = np.array([0.0, 0.5, 1.0]), np.geomspace(1e-3, 50.0, 60)
    boost = nonlinear.boost(z, k, k_hunit=True)
    # no boost on large scales, P_NL > P_L on small scales, growing with time
    assert_allclose(boost[:, 0], 1.0, atol=1e-2)
    assert np.all(boost[:, -1] > 10.0)
    assert np.all(np.diff(boost[:, -1]) < 0.0)
    assert_allclose(
        nonlinear.matter_power_spectrum_cb(z, k, hubble_units=True, k_hunit=True),
        camb_linear.matter_power_spectrum_cb(z, k, hubble_units=True, k_hunit=True) * boost,
    )
    ms = MatterStatistics(
        camb_linear,
        z=np.linspace(1e-5, 2.0 - 1e-5, 50),
        k=np.geomspace(1e-4, 50.0, 400),
        nonlinear_perturbations=nonlinear,
    )
    R, zz = np.geomspace(0.3, 10.0, 5), np.array([0.5])
    ratio = ms.excess_surface_mass_density_2h(R, zz, nonlinear=True) / ms.excess_surface_mass_density_2h(R, zz)
    assert np.all(np.diff(ratio[0]) < 0.0) and ratio[0, -1] > 1.0


if __name__ == "__main__":
    test_emulator_perturbations()
    test_nonlinear_boost_perturbations()
