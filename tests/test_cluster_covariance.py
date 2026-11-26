# import jax.numpy as np
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.halos.halo_abundance import CastroHaloAbundance
from cloelib.observables.halos.halo_mass_observable import (
    LognormalPowerLawHaloMassObservable,
)
from cloelib.observables.halos.halo_model_properties import HaloModelProperties
from cloelib.observables.halos.halo_profile import NFWHaloProfile
from cloelib.observables.halos.selection_function import GaussianSelectionFunction
from cloelib.summary_statistics.clusters import (
    ClusterCounts,
    ClusterCovariance,
    ClusterStatisticsModeling,
    ClusterWeakLensing,
    ClusterWeakLensingCovariance,
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


def test_gt_covariance():

    # Gaussian covariance of the reduced shear profile
    print("# g_t covariance")

    _H0 = 67.7
    _h = _H0 / 100.0
    background = CAMBBackground(
        H0=_H0,
        Omega_cdm0=0.12 / _h**2,
        Omega_b0=0.022 / _h**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.06,
        As=2e-9,
        gamma_MG=0.0,
        N_mnu=1,
    )
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))
    halo_model_properties = HaloModelProperties(
        perturbations, z=np.linspace(0.01, 2.0, 100), k=np.geomspace(1e-4, 10.0, 500)
    )
    selection_function = GaussianSelectionFunction(
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            A_l=52.0, B_l=0.9, C_l=0.5, sig_A_l=0.2, sig_B_l=-0.05, sig_C_l=0.001
        ),
        sig_lambda_norm=0.9,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.4,
        sig_z_z=0.025,
        sig_z_lambda=5.0e-6,
        lambda_tab_integ=[31, 31],
        z_tab_integ=31,
    )
    cluster_statistics_modeling = ClusterStatisticsModeling(
        CastroHaloAbundance(halo_model_properties),
        selection_function,
        integ_k_arr=np.geomspace(1.1e-4, 9.9, 400),
        integ_mass_arr=np.logspace(12.0, 16.0, 121),
        integ_lambda_true_arr=np.geomspace(5.0, 250.0, 51),
        integ_ztrue_arr=np.linspace(0.1, 1.1, 101),
        area=10313.0,
    )
    profile = NFWHaloProfile(halo_model_properties, two_halo="None", zs_max=2.0)
    cluster_wl = ClusterWeakLensing(
        cluster_statistics_modeling, profile, halo_concentration=4.0
    )
    cluster_wl_cov = ClusterWeakLensingCovariance(cluster_wl, n_eff=15.0, sigma_e=0.3)

    z_obs_edges = np.array([0.3, 0.5, 0.7])
    lambda_obs_edges = np.array([20.0, 30.0, 60.0])
    radius_edges = np.geomspace(0.3, 10.0, 6)
    z_size, lambda_size, radius_size = 2, 2, 5

    cov, terms = cluster_wl_cov.get_gt_covariance(
        z_obs_edges, lambda_obs_edges, radius_edges, return_terms=True
    )

    print("    shape and symmetries")
    assert cov.shape == (z_size, z_size, lambda_size, lambda_size) + (radius_size,) * 2
    assert_allclose(cov, terms["shape"] + terms["lss"] + terms["intr"], rtol=1e-12)
    # diagonal in z_obs bins
    assert_equal(cov[0, 1], 0.0)
    assert_equal(cov[1, 0], 0.0)
    # symmetric and positive definite
    cov_2d = cov.transpose(0, 2, 4, 1, 3, 5).reshape(
        (z_size * lambda_size * radius_size,) * 2
    )
    assert_allclose(cov_2d, cov_2d.T, rtol=1e-12)
    assert np.linalg.eigvalsh(cov_2d).min() > 0.0

    print("    auto richness blocks")
    cov_auto = cluster_wl_cov.get_gt_covariance(
        z_obs_edges, lambda_obs_edges, radius_edges, cross_richness=False
    )
    for ind_lambda in range(lambda_size):
        assert_allclose(
            cov_auto[:, :, ind_lambda, ind_lambda],
            cov[:, :, ind_lambda, ind_lambda],
            rtol=1e-12,
        )
    assert_equal(cov_auto[:, :, 0, 1], 0.0)
    assert np.abs(cov[:, :, 0, 1]).max() > 0.0

    print("    intermediate products of get_gt")
    _, products = cluster_wl.get_gt(
        z_obs_edges,
        lambda_obs_edges,
        radius_edges,
        return_intermediate_products=True,
    )
    assert_allclose(
        cluster_wl_cov.get_gt_covariance(
            z_obs_edges,
            lambda_obs_edges,
            radius_edges,
            intermediate_integration_products=products,
        ),
        cov,
        rtol=1e-12,
    )

    print("    comparison with external values")
    # independent implementation of Wu et al. 2019 (notebook
    # "Test Covariance lensing gammat.ipynb")
    cov_reference = np.array(
        [
            [
                [
                    [2.386095e-08, 1.051288e-10, 2.522785e-11, 6.788929e-12, 1.593875e-12],
                    [1.051288e-10, 6.052341e-09, 1.029631e-10, 2.707710e-11, 6.620852e-12],
                    [2.522785e-11, 1.029631e-10, 1.675003e-09, 1.086223e-10, 2.691645e-11],
                    [6.788929e-12, 2.707710e-11, 1.086223e-10, 6.078572e-10, 1.090248e-10],
                    [1.593875e-12, 6.620852e-12, 2.691645e-11, 1.090248e-10, 3.420539e-10],
                ],
                [
                    [4.232912e-08, 2.001901e-10, 4.761521e-11, 1.165190e-11, 2.882675e-12],
                    [2.001901e-10, 1.076463e-08, 1.949587e-10, 4.829547e-11, 1.163091e-11],
                    [4.761521e-11, 1.949587e-10, 2.995371e-09, 1.958930e-10, 4.754604e-11],
                    [1.165190e-11, 4.829547e-11, 1.958930e-10, 1.088117e-09, 1.932811e-10],
                    [2.882675e-12, 1.163091e-11, 4.754604e-11, 1.932811e-10, 6.076511e-10],
                ],
            ],
            [
                [
                    [2.491300e-08, 6.730479e-11, 1.637752e-11, 4.506238e-12, 1.263327e-12],
                    [6.730479e-11, 6.241899e-09, 6.673842e-11, 1.876897e-11, 4.804217e-12],
                    [1.637752e-11, 6.673842e-11, 1.657551e-09, 7.470229e-11, 1.963755e-11],
                    [4.506238e-12, 1.876897e-11, 7.470229e-11, 5.440010e-10, 7.891225e-11],
                    [1.263327e-12, 4.804217e-12, 1.963755e-11, 7.891225e-11, 2.748715e-10],
                ],
                [
                    [4.893200e-08, 1.374554e-10, 3.286710e-11, 8.801758e-12, 2.254175e-12],
                    [1.374554e-10, 1.226987e-08, 1.343968e-10, 3.672035e-11, 9.233604e-12],
                    [3.286710e-11, 1.343968e-10, 3.262167e-09, 1.464453e-10, 3.780413e-11],
                    [8.801758e-12, 3.672035e-11, 1.464453e-10, 1.068587e-09, 1.522500e-10],
                    [2.254175e-12, 9.233604e-12, 3.780413e-11, 1.522500e-10, 5.335328e-10],
                ],
            ],
        ]
    )
    cov_auto_blocks = np.array(
        [
            [cov[iz, iz, il, il] for il in range(lambda_size)]
            for iz in range(z_size)
        ]
    )
    variance = np.diagonal(cov_auto_blocks, axis1=2, axis2=3)
    variance_reference = np.diagonal(cov_reference, axis1=2, axis2=3)
    assert_allclose(variance, variance_reference, rtol=1e-2)

    correlation = cov_auto_blocks / np.sqrt(
        variance[..., :, np.newaxis] * variance[..., np.newaxis, :]
    )
    correlation_reference = cov_reference / np.sqrt(
        variance_reference[..., :, np.newaxis] * variance_reference[..., np.newaxis, :]
    )
    assert_allclose(correlation, correlation_reference, atol=2e-3)
