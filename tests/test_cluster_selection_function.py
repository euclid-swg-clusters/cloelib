# import jax.numpy as np

import os

import numpy as np
from numpy.testing import assert_allclose
from cloelib.observables.halos.halo_mass_observable import (
    LognormalPowerLawHaloMassObservable,
    ShiftedPoissonHaloMassObservable,
)
from cloelib.observables.halos.selection_function import (
    GaussianSelectionFunction,
    NumericalSelectionFunction,
    AnalyticSelectionFunction,
)

# get validation data
import urllib.request

SEL_FUNC_VAL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmark_values_sel_fun.py"
)
if not os.path.exists(SEL_FUNC_VAL_PATH):
    sf_val_url = "https://zenodo.org/records/21511551/files/benchmark_values_sel_fun.py"
    print(f"Downloading selection function validation file from {sf_val_url} ...")
    urllib.request.urlretrieve(sf_val_url, SEL_FUNC_VAL_PATH)
import benchmark_values_sel_fun


def _get_test_gaussian_sf():
    _lambda_true_dist_pars = dict(
        A_l=52.0,
        B_l=0.9,
        C_l=0.5,
        sig_A_l=0.2,
        sig_B_l=-0.05,
        sig_C_l=0.001,
    )
    _sel_pars = dict(
        sig_lambda_norm=0.1,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.1,
        sig_z_z=0.1,
        sig_z_lambda=0.1,
    )
    return GaussianSelectionFunction(
        **_sel_pars,
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            **_lambda_true_dist_pars
        ),
        lambda_tab_integ=[31, 31, 31, 51],
        z_tab_integ=31,
    )

def _get_test_analytic_sf():
    _lambda_true_dist_pars = dict(
        A_l=52.0,
        B_l=0.9,
        C_l=0.5,
        sig_A_l=0.2,
        sig_B_l=-0.05,
        sig_C_l=0.001,
    )
    _sel_pars = dict(
        # scatter in lambda_obs
        sig_lambda_norm=0.21,
        sig_lambda_z=0.0,
        sig_lambda_exponent=0.54,
        # mean of lambda_obs
        mu_lambda_norm=0.65,
        mu_lambda_z=0.056,
        mu_lambda_0=17.6,
        # exponential tail slope
        tau_lambda_norm=0.23,
        tau_lambda_z=0.0,
        tau_lambda_exponent= - 0.22,
        # projected cluster fraction
        fprj_lambda_norm=0.04,
        fprj_lambda_z=-0.03,
        fprj_lambda_exponent=0.02,
        # scatter in z_obs
        sig_z_lambda_exponent=-0.26,
        sig_z_lambda_norm=0.0186,
        sig_z_z_norm=0.006,
        # quadrature
        z_tab_integ=41,
        lambda_tab_integ=[51],
    )
    return AnalyticSelectionFunction(
        **_sel_pars,
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            **_lambda_true_dist_pars
        ),
    )



def test_shifted_poisson_scaling_relation():
    _lambda_true_dist_pars = dict(
        M_min_cen=10.**11.073,
        M_min_sat=10.**12.191,
        alpha=0.879,
        epsilon=0.955,
        sigma_lnltr=0.208,
    )
    halo_mass_observable=ShiftedPoissonHaloMassObservable(
        **_lambda_true_dist_pars
        )

    z_test = np.linspace(0.1,2.0,3)
    l_test = np.linspace(1.,300,5)
    M_test = np.array([5.0e13,1.0e14,1.0e15]) 

    _mean_l_given_M_z_ref = np.array([[ 12.43029063,  22.04335889, 160.41228422],
       [ 21.71343835,  39.13379127, 289.87948941],
       [ 30.79737845,  55.85747907, 416.56845029]])
    assert_allclose(
        halo_mass_observable._mean_richness(z_test[:,np.newaxis], M_test[np.newaxis,:]),
        _mean_l_given_M_z_ref,
        rtol=1e-05,
    )

    _pdf_richness_ref=np.array([[[9.31712100e-004, 1.14711919e-028, 6.32575647e-091,
         1.94982998e-169, 1.13155990e-258],
        [9.67464862e-005, 2.17378363e-013, 1.74149291e-051,
         1.95688543e-104, 1.70335514e-167],
        [3.15527361e-007, 6.32290921e-004, 1.08547001e-002,
         2.12307489e-003, 6.09230231e-006]],
       [[1.02518156e-004, 1.03563752e-013, 2.05956996e-052,
         5.04460672e-106, 9.02356502e-170],
        [1.21920167e-005, 8.83209590e-005, 1.84563306e-022,
         8.75846707e-052, 1.21736553e-089],
        [1.13664500e-007, 1.64515024e-005, 5.22810128e-004,
         3.75950249e-003, 6.29715817e-003]],
       [[2.79285980e-005, 1.64854977e-007, 5.60207210e-033,
         1.09427351e-071, 9.56465901e-120],
        [3.98030558e-006, 9.91679205e-003, 2.02201863e-011,
         7.06171058e-029, 3.19097596e-053],
        [6.66429303e-008, 2.61175346e-006, 4.88156709e-005,
         4.38317806e-004, 1.90420737e-003]]])
    assert_allclose(
        halo_mass_observable.pdf_richness(z_test, M_test, l_test),
        _pdf_richness_ref,
        rtol=1e-05,
    )


def test_gaussian_selectionfunction():
    print("# SelectionFunction")
    selection_function = _get_test_gaussian_sf()

    new_pars = dict(A_l=0.5, B_l=0.6, C_l=0.5, sig_A_l=0.1, sig_B_l=0.0, sig_C_l=0.0)
    for key, val in new_pars.items():
        setattr(selection_function.halo_mass_observable, key, val)

    # tests
    z_test = np.linspace(0.01, 1.0, 5)
    zob_test = np.linspace(0.1, 1.1, 5)
    M_test = 1.0e14
    l_test = np.logspace(0.0, 2.0, 5)
    lob_test = np.logspace(0.2, 2.2, 5)

    # Tests

    print("    lnrichness")
    _lnrichness_ref = [-1.533121, -1.423534, -1.3337, -1.257575, -1.191523]
    assert_allclose(
        selection_function.halo_mass_observable._mean_lnrichness(z_test, M_test),
        _lnrichness_ref,
        rtol=1e-05,
    )
    print("    scatter_lnrichness")
    assert_allclose(
        selection_function.halo_mass_observable.scatter_lnrichness(z_test, M_test),
        0.1,
        rtol=1e-05,
    )
    print("    prob_true_richness")
    assert_allclose(
        selection_function.halo_mass_observable.pdf_richness(z_test, M_test, l_test),
        0,
        atol=1e-10,
        rtol=1e-05,
    )
    print("    scatter_lambda_obs")
    _scatter_lambda_obs_ref = [0.101, 0.113324, 0.127151, 0.142666, 0.160074]
    assert_allclose(
        selection_function._scatter_lambda_obs(z_test[:, None], l_test[None, :])[0],
        _scatter_lambda_obs_ref,
        rtol=1e-05,
    )
    print("    scatter_z_obs")
    _scatter_z_obs_ref = [0.159489, 0.526937, 1.635393, 5.087122, 15.948932]
    assert_allclose(
        selection_function.scatter_z_obs(lob_test, z_test),
        _scatter_z_obs_ref,
        rtol=1e-5,
    )

    # Prob functions tests
    print("    prob_lambda_obs")
    _prob_lambda_obs_ref = [2.06231e-07, 0.00000e00, 0.00000e00, 0.00000e00, 0.00000e00]
    assert_allclose(
        selection_function._prob_lambda_obs(
            z_test[:, None, None], l_test[None, :, None], lob_test[None, None, :]
        )[0, 0],
        _prob_lambda_obs_ref,
        rtol=1e-05,
    )
    print("    prob_z_obs")
    _prob_z_obs_ref = [
        2.133197e00,
        7.240213e-01,
        2.365759e-01,
        7.777955e-02,
        2.497394e-02,
    ]
    assert_allclose(
        selection_function._prob_z_obs(
            zob_test[:, None, None], lob_test[None, :], z_test
        )[0, 0],
        _prob_z_obs_ref,
        rtol=5e-07,
    )


def _gen_gaussian_selcl_data(gaussian_sf, arrays):
    sel_cl_data = {"arrays": arrays}

    sel_cl_data["step_size"] = {
        "z_obs": (
            sel_cl_data["arrays"]["z_obs"][1:] - sel_cl_data["arrays"]["z_obs"][:-1]
        ).mean(),
        "lambda_obs": (
            sel_cl_data["arrays"]["lambda_obs"][1:]
            - sel_cl_data["arrays"]["lambda_obs"][:-1]
        ).mean(),
    }
    sel_cl_data["area_tile"] = np.ones(3)

    # tables
    def _prob_func(zob, lob, ztr, ltr, alpha):
        return (
            gaussian_sf._prob_lambda_obs(
                ztr[None, :, None], ltr[None, None, :], lob[:, None, None]
            )[None, None, ...]
            * gaussian_sf._prob_z_obs(zob[:, None, None], lob[None, :, None], ztr)[
                None, :, :, :, None
            ]
            * alpha[:, None, None, None, None]
        )

    sel_cl_data["tables"] = {
        "prob_lambda_z_obs": _prob_func(
            sel_cl_data["arrays"]["z_obs"],
            sel_cl_data["arrays"]["lambda_obs"],
            sel_cl_data["arrays"]["z_true"],
            sel_cl_data["arrays"]["lambda_true"],
            sel_cl_data["area_tile"],
        ),
        "completeness": np.ones(
            (
                sel_cl_data["area_tile"].size,
                sel_cl_data["arrays"]["z_true"].size,
                sel_cl_data["arrays"]["lambda_true"].size,
            )
        ),
        "purity": np.ones(
            (
                sel_cl_data["area_tile"].size,
                sel_cl_data["arrays"]["z_obs"].size,
                sel_cl_data["arrays"]["lambda_obs"].size,
            )
        ),
    }

    return sel_cl_data


def _gen_kde_selcl_data():
    pdf = benchmark_values_sel_fun.pdf  # shape: (n_ltr, n_ztr, n_lob, n_zob)
    purity = benchmark_values_sel_fun.purity  # shape: (n_lob, n_zob)
    arrays = benchmark_values_sel_fun.axes

    sel_cl_data = {"arrays": arrays}

    sel_cl_data["step_size"] = {
        "z_obs": np.diff(arrays["z_obs"]).mean(),
        "lambda_obs": np.diff(arrays["lambda_obs"]).mean(),
    }

    sel_cl_data["area_tile"] = np.ones(3)

    # (n_ltr, n_ztr, n_lob, n_zob) -> (n_zob, n_lob, n_ztr, n_ltr)
    prob = np.nan_to_num(pdf).transpose(3, 2, 1, 0)
    # -> (n_alpha, n_zob, n_lob, n_ztr, n_ltr)
    prob = np.repeat(prob[None, ...], sel_cl_data["area_tile"].size, axis=0)

    # purity (n_lob, n_zob) -> (n_alpha, n_zob, n_lob)
    pur = np.nan_to_num(purity).T[None, ...]  # (1, n_zob, n_lob)
    pur = np.repeat(pur, sel_cl_data["area_tile"].size, axis=0)

    sel_cl_data["tables"] = {
        "prob_lambda_z_obs": prob,
        "completeness": np.ones(
            (
                sel_cl_data["area_tile"].size,
                arrays["z_true"].size,
                arrays["lambda_true"].size,
            )
        ),
        "purity": pur,
    }

    return sel_cl_data


def test_interpolated_selectionfunction_unittest():
    test_arrays = {
        "z_true": np.linspace(0, 3, 31),
        "lambda_true": np.linspace(5, 300, 29),
        "z_obs": np.linspace(0.3, 1.5, 25),
        "lambda_obs": np.linspace(25, 200, 27),
    }
    print("Test with gaussian input data")
    gaussian_sf = _get_test_gaussian_sf()
    sel_cl_data = _gen_gaussian_selcl_data(gaussian_sf, test_arrays)

    sfn = NumericalSelectionFunction(
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            A_l=52.0,
            B_l=0.9,
            C_l=0.5,
            sig_A_l=0.2,
            sig_B_l=-0.05,
            sig_C_l=0.001,
        ),
        sel_cl_data=sel_cl_data,
        prob_contains_completeness=False,
        extrapolate=0,
    )

    # Prob functions tests
    interps = sfn._build_windows_interpolators(
        z_obs_edges=test_arrays["z_obs"][::3],
        lambda_obs_edges=test_arrays["lambda_obs"][::2],
    )

    # data generated using np.trapz integration
    ref_data = np.array(
        [
            [9.84882806e-05, 9.88277686e-05, 9.91574652e-05, 9.94773278e-05],
            [1.00696049e-04, 1.01043737e-04, 1.01381426e-04, 1.01709073e-04],
            [1.00696049e-04, 1.01043737e-04, 1.01381426e-04, 1.01709073e-04],
        ]
    )
    x = np.linspace(2.99, 3.01, 3)
    y = np.linspace(36.5, 36.7, 4)
    assert_allclose(interps[1][1](x, y), ref_data, atol=1e-06)


def test_interpolated_selectionfunction_compare_with_gauss():
    test_arrays = {
        "z_true": np.linspace(0, 3, 31),
        "lambda_true": np.linspace(5, 300, 29),
        "z_obs": np.linspace(0.3, 1.5, 50),
        "lambda_obs": np.linspace(25, 200, 270),
    }
    print("Test with gaussian input data")
    gaussian_sf = _get_test_gaussian_sf()
    sel_cl_data = _gen_gaussian_selcl_data(gaussian_sf, test_arrays)

    sfn = NumericalSelectionFunction(
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            A_l=52.0,
            B_l=0.9,
            C_l=0.5,
            sig_A_l=0.2,
            sig_B_l=-0.05,
            sig_C_l=0.001,
        ),
        sel_cl_data=sel_cl_data,
        prob_contains_completeness=False,
        extrapolate=0,
    )

    # the test below is not valid because:
    # 1. They are not exactly the same distributions:
    # In wf_g, P(Dzob|lob,ztr) is computed at lob = lambda_obs_edges[:-1]
    # In wf_n, P(Dzob|lob,ztr) is computed at lob = test_arrays["lambda_obs"]
    # The integrations over lob and zob is performed over different grids
    # in the two cases. The values set here are chosen to match the number
    # of point used for integration in sfn, but not the exact grid values

    gaussian_sf.lambda_tab_integ = [10]
    gaussian_sf.z_tab_integ = 12

    wf_g = gaussian_sf.window_redshift_richness_observed(
        z_obs_edges=np.array([0.5, 0.8]),
        lambda_obs_edges=np.array([25.0, 31.50557620817844]),
        mass=1e14 * np.ones(3),
        z_true=np.linspace(0, 3, 31),
        lambda_true=np.linspace(5, 300, 29),
    )
    wf_n = sfn.window_redshift_richness_observed(
        z_obs_edges=np.array([0.5, 0.8]),
        lambda_obs_edges=np.array([25.0, 31.50557620817844]),
        mass=1e14 * np.ones(3),
    )

    assert_allclose(wf_g, wf_n, atol=2.5e-3)

def test_analytical_selectionfunction_compare_with_tabulated_values():
    
    sf = _get_test_analytic_sf()

    z_obs_edges      = np.array([0.5, 0.8])
    lambda_obs_edges = np.array([40., 55.])
    z_true           = np.arange(0.05, 1.05, 0.1)
    lambda_true      = np.geomspace(5, 300, 50)
    mass             = np.array([5.0e13, 1.0e14, 5.0e14])

    # --- test window_redshift_richness_observed_given_lambda_true ---
    wf_ltr = sf.window_redshift_lambda_observed(
        z_obs_edges=z_obs_edges,
        lambda_obs_edges=lambda_obs_edges,
        z_true=z_true,
        lambda_true=lambda_true,
    )
    # shape: (z_obs_bins, lambda_obs_bins, z_true, lambda_true) = (1, 1, 10, 50)
    assert wf_ltr.shape == (1, 1, z_true.size, lambda_true.size), (
        f"unexpected shape {wf_ltr.shape}"
    )

    # reference values at fixed (z_obs_bin=0, lambda_obs_bin=0, z_true[::3], lambda_true[::10])
    wf_ltr_ref = np.array([[5.39475954e-060, 1.33203968e-137, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000],
       [1.07678194e-005, 2.06252565e-006, 2.17363812e-011,
        8.09177270e-052, 1.00565778e-193],
       [5.46648627e-005, 4.37501366e-004, 3.03891514e-003,
        8.95067306e-001, 1.25435837e-044],
       [2.35447974e-005, 6.65519003e-005, 5.69578591e-006,
        5.95520128e-028, 9.69293555e-205]])
    assert_allclose(wf_ltr[0, 0, ::3, ::10], wf_ltr_ref, atol=1e-6)

    # sanity check: values should be in [0,1]
    assert np.all(wf_ltr >= 0.), "window_redshift_lambda_observed has negative values"

    # --- test window_redshift_richness_observed ---
    wf = sf.window_redshift_richness_observed(
        z_obs_edges=z_obs_edges,
        lambda_obs_edges=lambda_obs_edges,
        z_true=z_true,
        mass=mass,
        lambda_true=None,
    )
    # shape: (z_obs_bins, lambda_obs_bins, z_true, mass) = (1, 1, 10, 3)
    assert wf.shape == (1, 1, z_true.size, mass.size), (
        f"unexpected shape {wf.shape}"
    )

    # reference values at fixed (z_obs_bin=0, lambda_obs_bin=0)
    # i.e. wf[0, 0, :, :] shape (z_true, mass) — replace with your actual reference values
    wf_ref = np.array([[7.61599845e-036, 4.25536494e-045, 1.68165627e-110],
       [4.25648499e-006, 2.61380513e-007, 2.64930039e-020],
       [4.94060244e-004, 2.07331459e-003, 2.71621975e-001],
       [5.74158191e-005, 2.31162812e-005, 7.68118646e-015]]
    )

    assert_allclose(wf[0,0,::3,:], wf_ref, atol=1.0e-5)

    # sanity check: values should be non-negative
    assert np.all(wf >= 0.), "window_redshift_richness_observed has negative values"

def test_interpolated_selectionfunction_compare_with_tabulated_values():
    sel_cl_data = _gen_kde_selcl_data()

    sfn = NumericalSelectionFunction(
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            A_l=52.0,
            B_l=0.9,
            C_l=0.5,
            sig_A_l=0.2,
            sig_B_l=-0.05,
            sig_C_l=0.001,
        ),
        sel_cl_data=sel_cl_data,
        prob_contains_completeness=True,
        extrapolate=0,
    )

    wf_n = sfn._window_redshift_richness_observed_by_lambda_true(
        z_obs_edges=np.array([0.5, 0.8]),
        lambda_obs_edges=np.array([40.0, 55]),
    )

    # This reference table has been derived using np.trapz integration
    wf_n_ref = np.array(
        [
            3.78689011e-002,
            3.48183571e-002,
            3.75766486e-002,
            3.92520267e-002,
            4.04592213e-002,
            4.01584929e-002,
            4.03228467e-002,
            3.70730778e-002,
            3.96778163e-002,
            3.44967198e-002,
            1.20300962e-002,
            6.27394246e-003,
            6.44266829e-003,
            7.49759124e-003,
            8.82985022e-003,
            1.09760800e-002,
            1.33057074e-002,
            1.71079843e-002,
            2.17445582e-002,
            2.52312924e-002,
            3.32092339e-002,
            6.01847462e-002,
            2.09219773e-001,
            4.15255745e-001,
            4.61806408e-001,
            4.52948937e-001,
            4.10517513e-001,
            2.08924552e-001,
            1.93419278e-002,
            3.19196968e-004,
            9.01215609e-007,
            5.13403916e-012,
            1.64675616e-018,
            1.88911123e-028,
            7.92063627e-037,
            1.02070291e-048,
            4.38518777e-080,
            2.80603011e-109,
            1.61700736e-117,
            1.72669439e-136,
            3.62859999e-169,
            9.56626556e-189,
            1.61665465e-189,
            1.81892272e-249,
            0.00000000e000,
            0.00000000e000,
            0.00000000e000,
            0.00000000e000,
            0.00000000e000,
            0.00000000e000,
        ]
    )

    assert_allclose(wf_n_ref, wf_n[0, 0, 5, :], atol=1e-6)

    wf_n = sfn.window_redshift_richness_observed(
        z_obs_edges=np.array([0.5, 0.8]),
        lambda_obs_edges=np.array([40.0, 55]),
        mass=np.array([5.0e13, 1.0e14, 5.0e14]),
    )

    # This reference table has been derived using np.trapz integration
    wf_n_ref = np.array(
        [
            [0.00000000e000, 0.00000000e000, 0.00000000e000],
            [0.00000000e000, 0.00000000e000, 0.00000000e000],
            [2.10356489e-184, 6.74638288e-184, 1.07468437e-200],
            [1.28865124e-037, 4.38458610e-037, 2.92360577e-055],
            [8.49384996e-005, 4.48745335e-004, 2.30613091e-008],
            [4.66771355e-003, 2.98546730e-002, 1.32256914e-001],
            [1.14837685e-002, 6.13888917e-002, 2.13159523e-001],
            [6.96396717e-003, 3.21752869e-002, 8.52152292e-002],
            [1.43313971e-004, 5.77231079e-004, 1.44616350e-009],
            [1.43436280e-033, 4.60646225e-033, 1.27650061e-050],
        ]
    )

    assert_allclose(wf_n_ref, wf_n[0, 0, :, :], atol=1e-6)

def test_analytical_selectionfunction_window_lambda_observed():

    sf = _get_test_analytic_sf()
    # large projected fraction, with a power law that exceeds 1 at high richness
    sf.fprj_lambda_norm, sf.fprj_lambda_z, sf.fprj_lambda_exponent = 0.1, 0.2, 0.5

    lambda_obs_edges = np.array([20., 40., 60., 90., 150.])
    z_true = np.array([0.3, 0.7])
    lambda_true = np.array([10., 30., 60., 120.])

    f_prj = sf._fprj_lambda_obs(z_true[:, None], lambda_true[None, :])
    assert np.all(f_prj <= 1.0) and np.any(f_prj == 1.0)

    # closed form vs numerical integral of the PDF
    wf = sf.window_lambda_observed(lambda_obs_edges, z_true, lambda_true)
    wf_num = np.zeros_like(wf)
    for b in range(lambda_obs_edges.size - 1):
        lob = np.linspace(lambda_obs_edges[b], lambda_obs_edges[b + 1], 4001)
        pdf = sf._prob_lambda_obs(
            z_true[None, :, None], lambda_true[None, None, :], lob[:, None, None]
        ).astype(float)
        wf_num[b] = np.trapezoid(pdf, x=lob, axis=0)
    assert_allclose(wf, wf_num, atol=1e-6)


if __name__ == "__main__":
    print("test_gaussian_selectionfunction")
    test_gaussian_selectionfunction()

    print("Unit test")
    test_interpolated_selectionfunction_unittest()

    print("Comparison with Gaussian SF")
    test_interpolated_selectionfunction_compare_with_gauss()

    print("Comparison with Analytic SF")
    test_analytical_selectionfunction_compare_with_tabulated_values()

    print("Comparison with mock tabulated SF")
    test_interpolated_selectionfunction_compare_with_tabulated_values()

    print("Analytic SF: closed-form window in lambda_obs")
    test_analytical_selectionfunction_window_lambda_observed()
