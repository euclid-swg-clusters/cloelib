# import jax.numpy as np

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from scipy.stats import rayleigh

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.halo_abundance import CastroHaloAbundance
from cloelib.observables.clusters.halo_profile import (
    BMOHaloProfile,
    MiscBMOHaloProfile,
    MiscBMOHaloProfileEmu,
    NFWHaloProfile,
)
from cloelib.observables.clusters.halo_mass_observable import (
    LognormalPowerLawHaloMassObservable,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics
from cloelib.observables.clusters.selection_function import GaussianSelectionFunction
from cloelib.summary_statistics.clusters import (
    ClusterStatisticsModeling,
    ClusterWeakLensing,
    ClusterWeakLensingCovariance,
)


def _get_matter_statistics(**kwargs):
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

    return MatterStatistics(perturbations, **kwargs)


def _get_castro():
    return CastroHaloAbundance(_get_matter_statistics())


def test_array_shapes():

    # Profiles
    print("# Profiles ")
    _prof_kwargs = dict(
        two_halo="None",
        zs_max=2.0,
    )

    profile_nfw = NFWHaloProfile(_get_matter_statistics(), **_prof_kwargs)

    R_test = np.linspace(0.01, 1.0, 9)
    z_test = np.linspace(0.01, 0.5, 4)
    M_test = np.linspace(1e14, 5e14, 6)
    c_test = 4.0
    HS = _get_matter_statistics()
    profile = NFWHaloProfile(HS)

    _kwargs = {"R": R_test, "z": z_test, "M": M_test}
    out_shape = (z_test.size, M_test.size, R_test.size)

    for radius_units in ("Mpc/h", "radians", "degrees", "arcmin", "arcsec"):
        print(radius_units)
        _kwargs["radius_units"] = radius_units

        _r, _z, _m = profile.core.surface_mass_density_args(**_kwargs)
        assert (_r * _z * _m).shape == out_shape

        _kwargs["c"] = c_test

        _kwargs["halo_bias"] = np.ones((z_test.size, M_test.size))

        for two_halo in ("None", "sum", "max"):
            profile_nfw.two_halo = two_halo

            assert profile_nfw.surface_mass_density(**_kwargs).shape == out_shape
            assert profile_nfw.excess_surface_mass_density(**_kwargs).shape == out_shape

        _kwargs.pop("halo_bias")
        _kwargs.pop("c")


def _test_profile(profile, reference_vals):

    R_test = np.array([1])
    z_test = np.linspace(0.01, 0.5, 4)
    M_test = np.array([5e14])
    c_test = 4.0
    z_sources_test = np.linspace(0.6, 1, 5)
    mean_z_cl_test = np.linspace(0.2, 1.0, 4)
    HS = _get_matter_statistics()
    castro = _get_castro()
    halo_bias = castro.bias(z_test, M_test)

    print("    sigma_crit")
    assert_allclose(
        profile.core.sigma_crit(z_test, z_sources_test)[0],
        **reference_vals["sigma_crit"],
    )
    idx = np.array([profile.core._get_tomo_bin_index(mean_z_cl_test[i]) for i in range(mean_z_cl_test.size)])
    print("    n_zs_norM")
    n_zs_norMs = np.array([profile.core.n_zs_norM(mean_z_cl_test[i],idx[i]) for i in range(mean_z_cl_test.size)])
    assert_allclose(n_zs_norMs, **reference_vals["n_zs_norM"],atol=1.0e-5)
    print("    n_zs")
    nzs = np.array([profile.core.n_zs(z_sources_test,idx[i]) for i in range(mean_z_cl_test.size)])
    assert_allclose(nzs, **reference_vals["n_zs"],atol=1.0e-5)
    print("    surface_mass_density")
    assert_allclose(
        profile.surface_mass_density(R_test, z_test, M_test, c_test)[:, 0, 0],
        **reference_vals["surface_mass_density"],
    )
    print("    excess_surface_mass_density")
    assert_allclose(
        profile.excess_surface_mass_density(R_test, z_test, M_test, c_test)[:, 0, 0],
        **reference_vals["excess_surface_mass_density"],
    )
    print("    _surface_mass_density_2h")
    profile_2h = (
        profile.core.matter_statistics.surface_mass_density_2h(R_test, z_test)[
            :, np.newaxis, :
        ]
        * halo_bias[:, :, np.newaxis]
    )
    assert_allclose(
        profile_2h[:, 0, 0],
        **reference_vals["surface_mass_density_2h"],
    )
    print("    _excess_surface_mass_density_2h")
    profile_2h = (
        profile.core.matter_statistics.excess_surface_mass_density_2h(R_test, z_test)[
            :, np.newaxis, :
        ]
        * halo_bias[:, :, np.newaxis]
    )
    assert_allclose(
        profile_2h[:, 0, 0],
        **reference_vals["excess_surface_mass_density_2h"],
    )


def test_profiles():

    # Profiles
    print("# Profiles ")

    print("  NFW")
    _prof_kwargs = dict(
        two_halo="None",
        zs_max=2.0,
    )
    profile_nfw = NFWHaloProfile(_get_matter_statistics(), **_prof_kwargs)
    _reference_vals = {
        # All validation values have to be updated with extarnal values
        "sigma_crit": {
            "desired": [
                57269.705861,
                57134.890066,
                57034.779821,
                56957.599339,
                56896.34007,
            ],
            "rtol": 1e-5,
        },
        "n_zs_norM": {"desired": [1.12513478, 1.50243006, 1.28488508, 1.594103  ], "rtol": 1e-5},
        "n_zs": {
            "desired": [[0.83347014, 0.7820134 , 0.72356751, 0.66030692, 0.59431415],
       [0.83347014, 0.7820134 , 0.72356751, 0.66030692, 0.59431415],
       [0.29190244, 0.5992073 , 0.84143889, 0.92745609, 0.9132132 ],
       [0.08161356, 0.18759727, 0.34873945, 0.53502114, 0.69604214]],
            "rtol": 1e-5,
        },
        "surface_mass_density": {
            "desired": [57.782346, 60.278322, 62.62685, 64.791899],
            "rtol": 1e-3,
        },
        "excess_surface_mass_density": {
            "desired": [85.397604, 94.3122, 103.483482, 112.700292],
            "rtol": 1e-3,
        },
        "surface_mass_density_2h": {
            "desired": [13.316477, 16.810355, 20.655992, 24.778575],
            "rtol": 1e-3,
        },
        "excess_surface_mass_density_2h": {
            "desired": [1.520876, 2.16498, 2.853623, 3.655179],
            "rtol": 1e-3,
        },
    }
    _test_profile(profile_nfw, _reference_vals)

    print("  BMO")
    _reference_vals.update(
        # All validation values have to be updated with extarnal values
        {
            "surface_mass_density": {
                "desired": [49.754672, 50.309348, 50.479286, 50.276593],
                "rtol": 1e-3,
            },
            "excess_surface_mass_density": {
                "desired": [90.134602, 99.25608, 108.537932, 117.745383],
                "rtol": 1e-3,
            },
        }
    )

    _prof_kwargs = dict(
        two_halo="None",
        trunc_fact=3.0,
        zs_max=2.0,
    )
    profile_bmo = BMOHaloProfile(_get_matter_statistics(), **_prof_kwargs)
    _test_profile(profile_bmo, _reference_vals)


@pytest.mark.parametrize("profile_class", [MiscBMOHaloProfileEmu, MiscBMOHaloProfile])
def test_misc_bmo_profile(profile_class):

    # Miscentered BMO profile (emulator or numerical) with miscentered 2-halo term
    print("# Miscentered BMO profile (%s)" % profile_class.__name__)

    R_test = np.array([0.1, 0.3, 1.0, 3.0])  # Mpc/h, below and above sigma_off
    z_test = np.linspace(0.01, 0.5, 4)
    M_test = np.array([5e14])
    c_test = 4.0
    sigma_off_test = 0.2  # Mpc/h
    halo_bias = _get_castro().bias(z_test, M_test)

    _prof_kwargs = dict(
        two_halo="None",
        trunc_fact=3.0,
        zs_max=2.0,
    )
    # k up to 100 h/Mpc: the default k range (k < 10 h/Mpc) is not enough to
    # converge the 2-halo excess surface density at R < 1 Mpc/h
    profile = profile_class(
        _get_matter_statistics(k=np.geomspace(1e-4, 100.0, 1000)), **_prof_kwargs
    )

    # Reference values from the notebook
    # "Check Miscentering model for Sigma and DeltaSigma.ipynb" (1-halo: emulator).
    # Shape (z_test.size, R_test.size), units h * Msun / pc**2.
    # The 2-halo values refer to the semi-approximated functions
    # (*_2h_off_semiapprox); the 2-halo term included in the profile uses the exact
    # average over the offsets (*_2h_off), tested in test_miscentered_2h_convolution.
    # The 2-halo terms are unbiased (divided by the halo bias). Their tolerance
    # accounts for the accuracy of the k integration (quad_vec, epsrel=1e-1).
    _reference_vals = {
        "surface_mass_density_1h": {
            "desired": [
                [431.653054, 295.611906, 52.419616, 3.554117],
                [479.717595, 325.194688, 53.507305, 3.148945],
                [531.988558, 353.716848, 53.874628, 2.753409],
                [582.963035, 381.765723, 54.095692, 2.390876],
            ],
            "rtol": 1e-3,
        },
        "excess_surface_mass_density_1h": {
            "desired": [
                [12.205999, 71.193244, 83.252068, 22.898200],
                [14.048267, 81.693802, 92.114262, 24.005986],
                [16.073981, 93.118577, 100.776741, 24.936328],
                [18.144897, 104.940648, 109.466341, 25.723668],
            ],
            "rtol": 1e-3,
        },
        "surface_mass_density_2h_off": {
            "desired": [
                [4.003221, 3.847079, 3.085107, 2.021469],
                [4.469531, 4.277211, 3.358121, 2.117295],
                [4.805877, 4.580837, 3.525509, 2.143991],
                [5.038332, 4.784210, 3.613157, 2.123500],
            ],
            "rtol": 3e-3,
        },
        "excess_surface_mass_density_2h_off": {
            "desired": [
                [0.174915, 0.209994, 0.353412, 0.484904],
                [0.215827, 0.258248, 0.425028, 0.563453],
                [0.253030, 0.301783, 0.486686, 0.625086],
                [0.286278, 0.340393, 0.538682, 0.671786],
            ],
            "rtol": 3e-3,
        },
    }

    if profile_class is MiscBMOHaloProfile:
        # the reference values of the 1-halo term above are the emulator outputs; for the
        # numerical profile, brute-force average over the offsets and the angles
        # (Rayleigh offsets up to 10 sigma_off, 2000 x 400 points)
        _reference_vals["surface_mass_density_1h"] = {
            "desired": [
                [431.405463, 295.527607, 52.612983, 3.546122],
                [480.148749, 324.209863, 53.480648, 3.142945],
                [531.200480, 353.573375, 53.965613, 2.750463],
                [583.384234, 382.911211, 54.071096, 2.384888],
            ],
            "rtol": 1e-3,
        }
        _reference_vals["excess_surface_mass_density_1h"] = {
            "desired": [
                [11.570131, 70.902438, 83.372876, 22.922412],
                [13.338055, 81.271438, 91.955416, 23.962314],
                [15.260834, 92.462773, 100.720619, 24.898397],
                [17.298155, 104.232208, 109.447242, 25.714994],
            ],
            # 5e-3 at R = sigma_off / 2, where DeltaSigma is small
            "rtol": 6e-3,
        }

    Sigma_1h = profile.surface_mass_density(
        R_test, z_test, M_test, c_test, sigma_off_test
    )
    DeltaSigma_1h = profile.excess_surface_mass_density(
        R_test, z_test, M_test, c_test, sigma_off_test
    )
    Sigma_2h_semi = profile.core.surface_mass_density_2h_off_semiapprox(
        R_test, z_test, sigma_off_test
    )
    DeltaSigma_2h_semi = profile.core.excess_surface_mass_density_2h_off_semiapprox(
        R_test, z_test, sigma_off_test
    )
    Sigma_2h = profile.core.surface_mass_density_2h_off(R_test, z_test, sigma_off_test)
    DeltaSigma_2h = profile.core.excess_surface_mass_density_2h_off(
        R_test, z_test, sigma_off_test
    )

    # 2-halo inclusion
    for two_halo in ("sum", "max"):
        print("    two_halo = %s" % two_halo)
        profile.two_halo = two_halo
        combine = np.add if two_halo == "sum" else np.maximum

        assert_allclose(
            profile.surface_mass_density(
                R_test, z_test, M_test, c_test, sigma_off_test, halo_bias=halo_bias
            ),
            combine(Sigma_1h, Sigma_2h[:, np.newaxis, :] * halo_bias[:, :, np.newaxis]),
            rtol=1e-10,
        )
        assert_allclose(
            profile.excess_surface_mass_density(
                R_test, z_test, M_test, c_test, sigma_off_test, halo_bias=halo_bias
            ),
            combine(
                DeltaSigma_1h,
                DeltaSigma_2h[:, np.newaxis, :] * halo_bias[:, :, np.newaxis],
            ),
            rtol=1e-10,
        )

    # the 2-halo term needs the halo bias
    assert_raises(
        ValueError,
        profile.surface_mass_density,
        R_test,
        z_test,
        M_test,
        c_test,
        sigma_off_test,
    )

    # Comparison with external values
    _computed_vals = {
        "surface_mass_density_1h": Sigma_1h[:, 0, :],
        "excess_surface_mass_density_1h": DeltaSigma_1h[:, 0, :],
        "surface_mass_density_2h_off": Sigma_2h_semi,
        "excess_surface_mass_density_2h_off": DeltaSigma_2h_semi,
    }
    missing = [k for k, v in _reference_vals.items() if v["desired"] is None]
    for key, vals in _reference_vals.items():
        if vals["desired"] is None:
            continue
        print("    %s" % key)
        assert_allclose(_computed_vals[key], **vals)

    if missing:
        pytest.skip("Missing reference values: %s" % ", ".join(missing))


def test_gt_covariance():

    # Gaussian covariance of the reduced shear profile
    print("# g_t covariance")

    integ_ztrue_arr = np.linspace(0.1, 1.1, 101)
    integ_mass_arr = np.logspace(12.0, 16.0, 121)
    matter_stat = _get_matter_statistics(
        z=np.linspace(0.01, 2.0, 100), k=np.geomspace(1e-4, 10.0, 500)
    )
    cluster_statistics_modeling = ClusterStatisticsModeling(
        CastroHaloAbundance(matter_statistics=matter_stat),
        integ_k_arr=np.geomspace(1.1e-4, 9.9, 400),
        integ_mass_arr=integ_mass_arr,
        integ_lambda_true_arr=np.geomspace(5.0, 250.0, 51),
        integ_ztrue_arr=integ_ztrue_arr,
        area=10313.0,
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
    cluster_wl = ClusterWeakLensing(
        cluster_statistics_modeling,
        NFWHaloProfile(matter_stat, two_halo="None", zs_max=2.0),
        halo_concentration=4.0,
        selection_function=selection_function,
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

    print("    concentration-mass relation")
    # a constant relation gives the covariance of the constant concentration
    cluster_wl_cM = ClusterWeakLensing(
        cluster_statistics_modeling,
        cluster_wl.profile,
        halo_concentration=lambda z, M: np.full((z.size, M.size), 4.0),
        selection_function=selection_function,
    )
    assert_allclose(
        ClusterWeakLensingCovariance(cluster_wl_cM, n_eff=15.0, sigma_e=0.3).get_gt_covariance(
            z_obs_edges, lambda_obs_edges, radius_edges
        ),
        cov,
        rtol=1e-12,
    )

    # Reference values from the notebook "Test Covariance lensing gammat.ipynb"
    # (last cell), computed with the original implementation of Wu et al. 2019
    # Eq. 10 using the same cloelib ingredients.
    # Auto richness blocks, shape (z_obs, lambda_obs, radius, radius).
    # The variances are compared with relative tolerance, the off-diagonal
    # terms through the correlation matrix: the notebook integrates numerically
    # the lens shot noise x shape noise term, which leaks a spurious correlation
    # between adjacent radial bins of ~1e-4 (exactly zero analytically).
    print("    comparison with external values")
    cov_reference = np.array(
        [
            [
                [
                    [3.281672e-08, 7.077090e-11, 1.568597e-11, 4.359248e-12, 1.012907e-12],
                    [7.077090e-11, 8.189872e-09, 6.449401e-11, 1.706008e-11, 4.310390e-12],
                    [1.568597e-11, 6.449401e-11, 2.130659e-09, 6.830604e-11, 1.749422e-11],
                    [4.359248e-12, 1.706008e-11, 6.830604e-11, 6.478853e-10, 7.045661e-11],
                    [1.012907e-12, 4.310390e-12, 1.749422e-11, 7.045661e-11, 2.848028e-10],
                ],
                [
                    [5.820921e-08, 1.349009e-10, 2.970522e-11, 7.136410e-12, 1.878316e-12],
                    [1.349009e-10, 1.454609e-08, 1.229641e-10, 3.036858e-11, 7.509319e-12],
                    [2.970522e-11, 1.229641e-10, 3.795613e-09, 1.232244e-10, 3.082744e-11],
                    [7.136410e-12, 3.036858e-11, 1.232244e-10, 1.155562e-09, 1.247561e-10],
                    [1.878316e-12, 7.509319e-12, 3.082744e-11, 1.247561e-10, 5.054527e-10],
                ],
            ],
            [
                [
                    [3.260651e-08, 6.634224e-11, 1.543328e-11, 4.217002e-12, 1.229022e-12],
                    [6.634224e-11, 8.129639e-09, 6.326760e-11, 1.776314e-11, 4.561915e-12],
                    [1.543328e-11, 6.326760e-11, 2.115162e-09, 7.072966e-11, 1.869162e-11],
                    [4.217002e-12, 1.776314e-11, 7.072966e-11, 6.493981e-10, 7.505699e-11],
                    [1.229022e-12, 4.561915e-12, 1.869162e-11, 7.505699e-11, 2.940720e-10],
                ],
                [
                    [6.404261e-08, 1.350488e-10, 3.092210e-11, 8.222356e-12, 2.129840e-12],
                    [1.350488e-10, 1.597659e-08, 1.271718e-10, 3.471310e-11, 8.749552e-12],
                    [3.092210e-11, 1.271718e-10, 4.160022e-09, 1.383706e-10, 3.592380e-11],
                    [8.222356e-12, 3.471310e-11, 1.383706e-10, 1.274928e-09, 1.445602e-10],
                    [2.129840e-12, 8.749552e-12, 3.592380e-11, 1.445602e-10, 5.709630e-10],
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


def test_miscentered_2h_convolution():

    # Exact average of the 2-halo term over a Rayleigh distribution of the offsets
    print("# Miscentered 2-halo term (convolution)")

    matter_stat = _get_matter_statistics(k=np.geomspace(1e-4, 100.0, 1000))
    profile = BMOHaloProfile(matter_stat, two_halo="None", zs_max=2.0)
    R_test = np.array([0.1, 0.3, 1.0, 3.0, 10.0])  # Mpc/h
    z_test = np.array([0.3, 0.8])
    sigma_off = 0.2  # Mpc/h

    print("    Sigma vs brute-force average")
    Sigma_off = profile.core.surface_mass_density_2h_off(R_test, z_test, sigma_off)
    ln_roff = np.linspace(
        np.log(rayleigh.ppf(1.0e-4, scale=sigma_off)),
        np.log(rayleigh.ppf(1.0 - 1.0e-4, scale=sigma_off)),
        120,
    )
    roff = np.exp(ln_roff)
    theta = np.linspace(0.0, 2.0 * np.pi, 121)
    dist = np.sqrt(
        R_test[None, :, None] ** 2.0
        + roff[:, None, None] ** 2.0
        + 2.0 * R_test[None, :, None] * roff[:, None, None] * np.cos(theta)[None, None, :]
    )
    Sigma_dist = matter_stat.surface_mass_density_2h(dist.ravel(), z_test).reshape(
        (z_test.size,) + dist.shape
    )
    weights = (rayleigh.pdf(roff, scale=sigma_off) * roff)[None, :, None]
    Sigma_brute = np.trapezoid(
        np.trapezoid(Sigma_dist, theta, axis=-1) / (2.0 * np.pi) * weights,
        ln_roff,
        axis=1,
    )
    assert_allclose(Sigma_off, Sigma_brute, rtol=1e-3)

    print("    DeltaSigma for a small offset vs the centered J2 Hankel transform")
    DeltaSigma_off = profile.core.excess_surface_mass_density_2h_off(
        R_test, z_test, 0.005
    )
    assert_allclose(
        DeltaSigma_off,
        matter_stat.excess_surface_mass_density_2h(R_test, z_test),
        atol=2e-3 * matter_stat.surface_mass_density_2h(R_test, z_test).max(),
    )

    print("    angular radius units")
    D_A = matter_stat.angular_diameter_distance(z_test) * matter_stat.background.h
    R_arcmin = np.degrees(R_test[1] / D_A[0]) * 60.0 * np.ones(2)
    assert_allclose(
        profile.core.surface_mass_density_2h_off(
            R_arcmin, z_test[:1], sigma_off, radius_units="arcmin"
        )[0],
        Sigma_off[0, 1],
        rtol=1e-6,
    )


def test_lensing_miscentering():

    # Centered + miscentered profiles and optical selection bias correction
    print("# Lensing profile with miscentering")

    matter_stat = _get_matter_statistics(
        z=np.linspace(0.01, 2.0, 100), k=np.geomspace(1e-4, 10.0, 500)
    )
    cluster_statistics_modeling = ClusterStatisticsModeling(
        CastroHaloAbundance(matter_statistics=matter_stat),
        integ_k_arr=np.geomspace(1.1e-4, 9.9, 400),
        integ_mass_arr=np.logspace(13.5, 15.5, 41),
        integ_lambda_true_arr=np.geomspace(5.0, 250.0, 51),
        integ_ztrue_arr=np.linspace(0.2, 0.9, 15),
        area=10313.0,
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
    cluster_wl = ClusterWeakLensing(
        cluster_statistics_modeling,
        BMOHaloProfile(matter_stat, two_halo="sum", trunc_fact=3.0, zs_max=2.0),
        halo_concentration=4.0,
        selection_function=selection_function,
        miscentered_profile=MiscBMOHaloProfileEmu(
            matter_stat, two_halo="sum", trunc_fact=3.0, zs_max=2.0
        ),
    )
    z_obs_edges = np.array([0.3, 0.5, 0.7])
    lambda_obs_edges = np.array([20.0, 30.0, 60.0])
    radius_edges = np.geomspace(0.3, 10.0, 6)
    args = (z_obs_edges, lambda_obs_edges, radius_edges)
    sigma_off = 0.3  # Mpc/h

    DS_cen = cluster_wl.get_DeltaSigma(*args)
    DS_cen_mix = cluster_wl.get_DeltaSigma(*args, miscentering_params=(1.0, sigma_off))
    DS_off = cluster_wl.get_DeltaSigma(*args, miscentering_params=(0.0, sigma_off))
    DS_mix = cluster_wl.get_DeltaSigma(*args, miscentering_params=(0.7, sigma_off))

    print("    centered and miscentered fractions")
    assert DS_mix.shape == (2, 2, 5)
    assert_allclose(DS_cen_mix, DS_cen, rtol=1e-12)
    assert_allclose(DS_mix, 0.7 * DS_cen + 0.3 * DS_off, rtol=1e-10)
    # the miscentering lowers DeltaSigma at small radii
    assert np.all(DS_off[..., 0] < DS_cen[..., 0])

    print("    mixture of miscentered populations")
    assert_allclose(
        cluster_wl.get_DeltaSigma(*args, miscentering_params=((0.7, None), (0.3, sigma_off))),
        DS_mix,
        rtol=1e-12,
    )
    DS_off_narrow = cluster_wl.get_DeltaSigma(*args, miscentering_params=(0.0, 0.1))
    assert_allclose(
        cluster_wl.get_DeltaSigma(*args, miscentering_params=((0.4, 0.1), (0.6, sigma_off))),
        0.4 * DS_off_narrow + 0.6 * DS_off,
        rtol=1e-10,
    )

    print("    concentration-mass relation")
    # constant relation = constant concentration
    cluster_wl_cM = ClusterWeakLensing(
        cluster_statistics_modeling,
        cluster_wl.profile,
        halo_concentration=lambda z, M: np.full((z.size, M.size), 4.0),
        selection_function=selection_function,
        miscentered_profile=cluster_wl.miscentered_profile,
    )
    assert_allclose(
        cluster_wl_cM.get_DeltaSigma(*args, miscentering_params=(0.7, sigma_off)),
        DS_mix,
        rtol=1e-12,
    )
    # profiles with a concentration for each (z, M)
    R_test, z_test, M_test = np.geomspace(0.3, 5.0, 4), np.array([0.3, 0.6]), np.array([1.0e14, 5.0e14])
    c_test = np.array([[4.0, 5.0], [3.0, 6.0]])
    # (the emulator is evaluated in single precision: rounding depends on the batch size)
    for prof, extra, rtol in [
        (BMOHaloProfile(matter_stat, trunc_fact=3.0, zs_max=2.0), (), 1e-10),
        (MiscBMOHaloProfileEmu(matter_stat, trunc_fact=3.0, zs_max=2.0), (sigma_off,), 1e-5),
        (MiscBMOHaloProfile(matter_stat, trunc_fact=3.0, zs_max=2.0), (sigma_off,), 1e-10),
    ]:
        for func in (prof.surface_mass_density, prof.excess_surface_mass_density):
            out = func(R_test, z_test, M_test, c_test, *extra)
            for i in range(2):
                for j in range(2):
                    assert_allclose(
                        out[i, j],
                        func(R_test, z_test[i : i + 1], M_test[j : j + 1], c_test[i, j], *extra)[0, 0],
                        rtol=rtol,
                    )

    print("    comoving radii")
    # with (almost) a single true redshift, comoving radii R are physical radii R / (1 + z)
    cluster_wl_z = ClusterWeakLensing(
        ClusterStatisticsModeling(
            CastroHaloAbundance(matter_statistics=matter_stat),
            integ_k_arr=np.geomspace(1.1e-4, 9.9, 400),
            integ_mass_arr=np.logspace(13.5, 15.5, 41),
            integ_lambda_true_arr=np.geomspace(5.0, 250.0, 51),
            integ_ztrue_arr=np.array([0.5, 0.5001]),
            area=10313.0,
        ),
        cluster_wl.profile,
        halo_concentration=4.0,
        selection_function=selection_function,
        miscentered_profile=cluster_wl.miscentered_profile,
    )
    args_z = (np.array([0.4, 0.6]), lambda_obs_edges)
    assert_allclose(
        cluster_wl_z.get_DeltaSigma(
            *args_z, radius_edges, miscentering_params=(0.7, sigma_off), comoving_radii=True
        ),
        cluster_wl_z.get_DeltaSigma(
            *args_z, radius_edges / 1.50005, miscentering_params=(0.7, sigma_off)
        ),
        rtol=2e-4,
    )

    print("    optical selection bias correction")
    corr_pars = (1.2, 0.2, 4.0, -0.3, 1.6)
    gt_mix = cluster_wl.get_gt(*args, miscentering_params=(0.7, sigma_off))
    gt_corr = cluster_wl.get_gt(
        *args, opt_sel_bias_params=corr_pars, miscentering_params=(0.7, sigma_off)
    )
    R_mean = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    assert_allclose(
        gt_corr,
        gt_mix * cluster_wl.optical_selection_bias_correction(R_mean, *corr_pars),
        rtol=1e-12,
    )
    # the default offset is 1
    assert_allclose(
        cluster_wl.get_gt(
            *args,
            opt_sel_bias_params=corr_pars + (1.0,),
            miscentering_params=(0.7, sigma_off),
        ),
        gt_corr,
        rtol=1e-12,
    )
    assert_allclose(
        cluster_wl.get_gt(
            *args,
            opt_sel_bias_params=corr_pars + (1.1,),
            miscentering_params=(0.7, sigma_off),
        ),
        gt_corr + 0.1 * gt_mix,
        rtol=1e-10,
    )

    print("    PCA form of the optical selection bias correction")
    rng = np.random.default_rng(3)
    B_mean = 1.0 + 0.1 * np.log(R_mean)
    modes = np.linalg.qr(rng.standard_normal((R_mean.size, 2)))[0].T
    amplitudes = rng.normal(0.0, 0.1, (2, 2, 2))
    gt_pca = cluster_wl.get_gt(
        *args,
        miscentering_params=(0.7, sigma_off),
        opt_sel_bias_pca=(B_mean, modes, amplitudes),
    )
    corr_pca = cluster_wl.optical_selection_bias_pca_correction(B_mean, modes, amplitudes)
    assert corr_pca.shape == (2, 2, 5)
    assert_allclose(corr_pca[1, 0], B_mean + amplitudes[1, 0] @ modes, rtol=1e-12)
    assert_allclose(gt_pca, gt_mix * corr_pca, rtol=1e-12)
    # templates can also depend on the bin
    assert_allclose(
        cluster_wl.optical_selection_bias_pca_correction(
            np.broadcast_to(B_mean, (2, 2, 5)), np.broadcast_to(modes, (2, 2, 2, 5)), amplitudes
        ),
        corr_pca,
        rtol=1e-12,
    )
    # covariance from the amplitudes: block diagonal, equal to the scatter of the
    # corrected profile for Gaussian amplitudes
    amp_cov = np.einsum("zlij,zlkj->zlik", *(2 * [rng.normal(0.0, 0.05, (2, 2, 2, 2))]))
    cov_pca = cluster_wl.optical_selection_bias_pca_covariance(gt_mix, modes, amp_cov)
    assert cov_pca.shape == (2, 2, 2, 2, 5, 5)
    assert np.all(cov_pca[0, 1] == 0.0) and np.all(cov_pca[:, :, 0, 1] == 0.0)
    samples = np.array(
        [
            gt_mix[1, 1]
            * cluster_wl.optical_selection_bias_pca_correction(B_mean, modes, a)
            for a in rng.multivariate_normal(amplitudes[1, 1], amp_cov[1, 1], 40000)
        ]
    )
    assert_allclose(np.cov(samples.T), cov_pca[1, 1, 1, 1], rtol=0.05, atol=1e-3 * np.max(cov_pca))

    print("    reduced shear")
    gt_lin = cluster_wl.get_gt(
        *args, miscentering_params=(0.7, sigma_off), reduced_shear=False
    )
    ratio = gt_mix / gt_lin
    # g_t = gamma_t / (1 - kappa) > gamma_t, with kappa decreasing with radius
    assert np.all(ratio > 1.0) and np.all(ratio < 1.5)
    assert np.all(np.diff(ratio, axis=-1) < 0.0)
    # with <Sigma_crit^-2> = 0 the reduced shear is the tangential shear
    sig_crit_inv = np.array(
        [
            cluster_wl.profile.core.sigma_crit_inv_eff(
                cluster_statistics_modeling.tabulated_integrands["ztrue"],
                cluster_wl.profile.core._get_tomo_bin_index(z),
            )
            for z in 0.5 * (z_obs_edges[:-1] + z_obs_edges[1:])
        ]
    )
    assert_allclose(
        cluster_wl._get_profile(
            *args,
            sig_crit_inv,
            miscentering_params=(0.7, sigma_off),
            effective_inverse_critical_surface_mass_density_2=np.zeros_like(
                sig_crit_inv
            ),
        ),
        gt_lin,
        rtol=1e-10,
    )

    print("    cluster contamination")
    # depends on redshift, richness and radial bin
    f_cl = 0.2 * np.exp(-R_mean) * np.array([[1.0, 1.5], [0.5, 2.0]])[..., np.newaxis]
    assert f_cl.shape == gt_mix.shape
    assert_allclose(
        cluster_wl.get_gt(
            *args, miscentering_params=(0.7, sigma_off), cluster_contamination=f_cl
        ),
        gt_mix * (1.0 - f_cl),
        rtol=1e-12,
    )
    # together with the optical selection bias correction
    assert_allclose(
        cluster_wl.get_gt(
            *args,
            opt_sel_bias_params=corr_pars,
            miscentering_params=(0.7, sigma_off),
            cluster_contamination=f_cl,
        ),
        gt_corr * (1.0 - f_cl),
        rtol=1e-12,
    )

    print("    errors")
    assert_raises(ValueError, cluster_wl.get_gt, *args, cluster_contamination=np.full(gt_mix.shape, 1.2))
    assert_raises(ValueError, cluster_wl.get_gt, *args, cluster_contamination=np.full(gt_mix.shape, -0.1))
    # wrong shapes: only the full (z_obs, lambda_obs, radius) array is accepted
    assert_raises(ValueError, cluster_wl.get_gt, *args, cluster_contamination=0.1)
    assert_raises(ValueError, cluster_wl.get_gt, *args, cluster_contamination=f_cl[0, 0])
    assert_raises(ValueError, cluster_wl.get_gt, *args, cluster_contamination=f_cl[..., 0])
    assert_raises(ValueError, cluster_wl.get_gt, *args, opt_sel_bias_params=corr_pars[:4])
    assert_raises(
        ValueError,
        cluster_wl.get_gt,
        *args,
        opt_sel_bias_params=corr_pars,
        opt_sel_bias_pca=(B_mean, modes, amplitudes),
    )
    assert_raises(ValueError, cluster_wl.get_DeltaSigma, *args, miscentering_params=(1.2, sigma_off))
    assert_raises(
        ValueError, cluster_wl.get_DeltaSigma, *args, miscentering_params=((0.5, None), (0.4, sigma_off))
    )
    cluster_wl_cen = ClusterWeakLensing(
        cluster_statistics_modeling,
        BMOHaloProfile(matter_stat, two_halo="sum", trunc_fact=3.0, zs_max=2.0),
        halo_concentration=4.0,
        selection_function=selection_function,
    )
    assert_raises(ValueError, cluster_wl_cen.get_DeltaSigma, *args, miscentering_params=(0.7, sigma_off))


def test_misc_bmo_profile_swap():

    # the numerical miscentered profile can replace the emulator in ClusterWeakLensing
    print("# Miscentered BMO profile: numerical vs emulator")
    matter_stat = _get_matter_statistics(
        z=np.linspace(0.01, 2.0, 100), k=np.geomspace(1e-4, 10.0, 500)
    )
    cluster_statistics_modeling = ClusterStatisticsModeling(
        CastroHaloAbundance(matter_statistics=matter_stat),
        integ_k_arr=np.geomspace(1.1e-4, 9.9, 400),
        integ_mass_arr=np.logspace(13.5, 15.5, 41),
        integ_lambda_true_arr=np.geomspace(5.0, 250.0, 51),
        integ_ztrue_arr=np.linspace(0.2, 0.9, 15),
        area=10313.0,
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
    args = (np.array([0.3, 0.5, 0.7]), np.array([20.0, 30.0, 60.0]), np.geomspace(0.3, 10.0, 6))
    gt = {}
    for profile_class in (MiscBMOHaloProfileEmu, MiscBMOHaloProfile):
        cluster_wl = ClusterWeakLensing(
            cluster_statistics_modeling,
            BMOHaloProfile(matter_stat, two_halo="sum", trunc_fact=3.0, zs_max=2.0),
            halo_concentration=lambda z, M: 7.85 * (M[np.newaxis, :] / 2.0e12) ** -0.081
            * (1.0 + z[:, np.newaxis]) ** -0.71,
            selection_function=selection_function,
            miscentered_profile=profile_class(matter_stat, two_halo="sum", trunc_fact=3.0, zs_max=2.0),
        )
        gt[profile_class] = cluster_wl.get_gt(
            *args, miscentering_params=((0.4, 0.1), (0.6, 0.3)), comoving_radii=True
        )
    # emulator accuracy ~1e-2
    assert_allclose(gt[MiscBMOHaloProfile], gt[MiscBMOHaloProfileEmu], rtol=1e-2)


@pytest.mark.parametrize("profile_class", [MiscBMOHaloProfileEmu, MiscBMOHaloProfile])
def test_misc_bmo_profile_angular_radii(profile_class):

    # angular radii correspond to physical radii depending on z
    print("# Miscentered BMO profile with angular radii (%s)" % profile_class.__name__)
    matter_stat = _get_matter_statistics()
    profile = profile_class(matter_stat, trunc_fact=3.0, zs_max=2.0)
    theta = np.geomspace(1.0, 20.0, 5)  # arcmin
    z_test, M_test = np.array([0.3, 0.6]), np.array([1.0e14, 5.0e14])
    D_A = matter_stat.angular_diameter_distance(z_test) * matter_stat.background.h
    for func in (profile.surface_mass_density, profile.excess_surface_mass_density):
        out = func(theta, z_test, M_test, 4.0, 0.15, radius_units="arcmin")
        for i in range(z_test.size):
            assert_allclose(
                out[i],
                func(np.radians(theta / 60.0) * D_A[i], z_test[i : i + 1], M_test, 4.0, 0.15)[0],
                rtol=1e-5,
            )
