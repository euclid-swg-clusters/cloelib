# import jax.numpy as np
import numpy as np
from numpy.testing import assert_allclose, assert_raises

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.halos.halo_abundance import CastroHaloAbundance
from cloelib.observables.halos.halo_mass_observable import (
    LognormalPowerLawHaloMassObservable,
)
from cloelib.observables.halos.halo_model_properties import HaloModelProperties
from cloelib.observables.halos.selection_function import GaussianSelectionFunction
from cloelib.summary_statistics.clusters import (
    ClusterCounts,
    ClusterCovariance,
    ClusterStatisticsModeling,
)


def test_count_covariance():
    # Cosmology parameters
    print("# Cosmology parameters")
    _H0 = 67.7
    _h = _H0 / 100.0
    _omch2 = 0.12
    _ombh2 = 0.022
    _cosmo_pars = dict(
        H0=_H0,
        Omega_cdm0=_omch2 / _h**2,
        Omega_b0=_ombh2 / _h**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.06,
        As=2e-9,
        gamma_MG=0.0,
        N_mnu=1,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))

    # counts covariance
    print("# counts covariance")

    area = 15000
    nbins_z = 10
    z_tab_integ = 31

    k_min = 1e-4
    k_max = 2e0
    k_div = 300

    zbins = np.linspace(0, 2, nbins_z + 1)
    k_test = np.geomspace(k_min, k_max, k_div)

    halo_model_properties = HaloModelProperties(
        perturbations,
        k=np.geomspace(k_min / 2, k_max * 2, k_div),
    )
    halo_abundance = CastroHaloAbundance(halo_model_properties)
    selection_function = GaussianSelectionFunction(
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            A_l=52.0,
            B_l=0.9,
            C_l=0.5,
            sig_A_l=0.2,
            sig_B_l=-0.05,
            sig_C_l=0.001,
        ),
        sig_lambda_norm=0.9,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.4,
        sig_z_z=0.025,
        sig_z_lambda=5.0e-6,
        z_tab_integ=z_tab_integ,
        lambda_tab_integ=[31],
    )
    clusterstatmod = ClusterStatisticsModeling(
        halo_abundance=halo_abundance,
        selection_function=selection_function,
        integ_k_arr=k_test,
        integ_mass_arr=np.geomspace(1e12, 1e16, 31),
        integ_lambda_true_arr=np.geomspace(5.0, 250.0, 31),
        integ_ztrue_arr=np.linspace(0.01, 1.99, 31),
        area=area,
    )
    CC = ClusterCovariance(ClusterCounts(clusterstatmod), z_tab_integ=z_tab_integ)
    # Exercise the unchanged internal survey-window kernel against reference values.
    CC.rint = np.zeros((nbins_z, len(k_test), CC.ell_max + 1))

    print("    Covariance coefficients")
    KL = CC._Kl_coeff()
    # All validation values have to be updated with extarnal values
    assert_allclose(
        KL[:5], [0.282095, 0.310942, 0.1095, -0.074565, -0.091053], rtol=5e-6
    )

    print("    Covariance window")
    iz = 0
    # All validation values have to be updated with extarnal values
    assert_allclose(
        CC._cov_window(iz, zbins[iz : iz + 2], KL)[0, :5],
        np.array([0.99959593, 0.99956826, 0.9995387, 0.99950711, 0.99947336]),
        rtol=1e-6,
    )

    print("    Counts covariance with richness-dependent photo-z scatter")
    z_obs_edges = np.linspace(0.2, 1.4, 4)
    lambda_obs_edges = np.array([20.0, 40.0, 200.0])
    lambda_mid = 0.5 * (lambda_obs_edges[1:] + lambda_obs_edges[:-1])
    selection_function.lambda_tab_integ = [31, 31]
    counts = ClusterCounts(clusterstatmod)
    prediction, products = counts.get_NC(z_obs_edges, lambda_obs_edges)

    def _get_cov(scatter_z_obs, prediction=prediction):
        # the scatter enters only in the spatial part, not in prediction and products
        selection_function.scatter_z_obs = scatter_z_obs
        return ClusterCovariance(counts, z_tab_integ=z_tab_integ).compute(
            intermediates=products,
            prediction=prediction,
            z_obs_edges=z_obs_edges,
            lambda_obs_edges=lambda_obs_edges,
        )

    # scatter decreasing with richness, infinite for lambda_obs = 0
    def _scatter(lambda_obs, z):
        return 0.1 * lambda_obs**-0.3 * np.ones_like(z)

    cov = _get_cov(_scatter)
    assert cov.shape == (3, 3, 2, 2)
    assert np.all(np.isfinite(cov))
    # symmetric and positive definite, larger than the shot noise
    cov_matrix = cov.transpose(0, 2, 1, 3).reshape(6, 6)
    assert_allclose(cov_matrix, cov_matrix.T, rtol=1e-12)
    assert np.all(np.linalg.eigvalsh(cov_matrix) > 0.0)
    assert np.all(np.diag(cov_matrix) > prediction.flatten())

    # the scatter is evaluated at the mean richness of each redshift bin
    lambda_mean = np.average(
        np.tile(lambda_mid, (3, 1)), weights=prediction, axis=1
    )
    assert np.all((lambda_mean > lambda_mid[0]) & (lambda_mean < lambda_mid[1]))
    cov_mean = _get_cov(lambda lambda_obs, z: _scatter(lambda_mean, z))
    assert_allclose(cov, cov_mean, rtol=1e-12)

    # redshift bin without clusters
    prediction_empty = prediction.copy()
    prediction_empty[1] = 0.0
    assert np.all(np.isfinite(_get_cov(_scatter, prediction_empty)))

    # the edges of the richness bins are needed
    with assert_raises(ValueError):
        ClusterCovariance(counts).compute(
            intermediates=products, prediction=prediction, z_obs_edges=z_obs_edges
        )
