#!/usr/bin/env python
"""
Mock cluster observables for simulation-based inference (SBI) tests, from the cloelib cluster model.

For each point of parameter space the script computes, with cloelib (branch ``cl_dev_costanzi``,
tag ``sbi-mocks-v1``):

* NC   : cluster number counts in bins of observed redshift and observed richness, shape (n_z, n_lambda);
* g_t  : stacked reduced tangential shear profiles in the same bins and in radial bins of
         comoving radius, shape (n_z, n_lambda, n_R);

then adds noise drawn from the cloelib covariances evaluated at the same parameter point:

* NC  : Poisson (shot noise) + sample variance (``ClusterCounts.get_NC_covariance``);
* g_t : shape noise + LSS + intrinsic (1-halo) scatter, Wu et al. 2019
        (``ClusterWeakLensingCovariance.get_gt_covariance``), correlated between the richness and
        radial bins of the same redshift bin, independent between redshift bins.
NC and g_t are taken as independent (no cross-covariance).

Model ("A1900" setup of the Euclid DR1 cluster tests: PICCOLO mock, 1900 deg2)
-------------------------------------------------------------------------------
* Cosmology: flat LCDM with one massive neutrino; linear P(k) from the Euclid DR1 CosmoPower-JAX
  emulators in cloelib (downloaded from Zenodo at the first use), nonlinear P_cb from the HMcode
  boost emulator (only used in the 2-halo term with halo exclusion). sigma8 is the one of the
  cold dark matter + baryon spectrum (the one that enters the halo mass function).
* Halo mass function and bias: Castro et al. (virial masses).
* Richness-mass relation: shifted Poisson HOD,
      <lambda_true | M, z> = 1 + [(M - M_min) / (M_1 - M_min)]^alpha [(1 + z) / (1 + z_piv)]^epsilon,
  with intrinsic (super-Poisson) scatter sigma_lnltr <lambda_sat>.
* Selection: cloelib ``AnalyticSelectionFunction`` (Costanzi et al. 2019 form): observed richness =
  Gaussian + projection tail (exponentially modified Gaussian) around the true richness, and
  Gaussian photo-z errors P(z_ob | z_true) whose width depends on richness. Its 15 parameters are
  fixed to the best fit to the PICCOLO mock (calibration file).
* Weak lensing: BMO profile (truncated NFW, truncation 3 r_vir) with Duffy et al. 2008 c(M, z),
  2-halo term with halo exclusion (``two_halo="sum_exclusion"``), reduced shear (non-weak shear
  correction), source redshift distribution of the cloelib profile core (Euclid-like tomography).
* Miscentring: two Rayleigh populations,
      P(r_off) = f Ray(r_off; s_1) + (1 - f) Ray(r_off; s_2),
  with s_1, s_2 constant in comoving Mpc/h (physical scale s / (1 + z)).
* Optical selection bias (the richness-selected clusters are not a random subsample of the halos
  of the same mass): multiplicative correction of the profiles,
      B(R) = Bbar(R) + a_1 e_1(R) + a_2 e_2(R),
  with the template Bbar, the PCA modes e_k and two amplitudes a_k per (z, lambda) bin calibrated
  on PICCOLO. The amplitudes are fixed to their calibration (their statistical uncertainty is
  added to the g_t covariance), and the systematic uncertainty of the calibration is described by
  3 parameters shared by all the bins (as B_SYS in the DR1 chains):
      B(R) - 1 = (1 + B_s) [Bbar(R) - 1 + sum_k (a_k + B_da<k>) e_k(R)].

Free parameters (see PARAMETERS below for fiducial values and priors)
---------------------------------------------------------------------
* cosmology          : Omega_m, sigma8 (h, Omega_b, n_s, m_nu fixed);
* richness-mass (HOD): log10_M1, alpha, epsilon, sigma_lnltr (M_min fixed);
* miscentring        : misc_f, misc_s1, misc_s2 (comoving Mpc/h);
* B(R) systematics   : B_s, B_da1, B_da2.
12 parameters in total. The fiducial values are the true values of the PICCOLO mock (cosmology and
richness-mass relation), the calibration of the A1900 setup (miscentring) and 0 (B systematics).
Cosmology and HOD parameters have the uniform priors of the A1900 chains; miscentring and B
systematics have Gaussian priors (nuisance parameters).

Usage
-----
    # 1000 points drawn from the prior, 1 noise realisation each, 8 processes
    python generate_cluster_mocks.py --n-points 1000 --ncpu 8 --output mocks_prior.npz

    # 200 noise realisations at the fiducial point (e.g. for coverage tests of the posteriors)
    python generate_cluster_mocks.py --mode fiducial --n-real 200 --output mocks_fiducial.npz

    # some parameters fixed to the fiducial (names or groups: cosmo, hod, misc, bsys)
    python generate_cluster_mocks.py --n-points 1000 --fix misc,bsys --output mocks_fixed_nuis.npz

    # your own parameter points (an .npy file with shape (n_points, n_params), in the order of
    # --list-params; or an .npz file with one array per parameter name, missing ones at the fiducial)
    python generate_cluster_mocks.py --mode file --theta-file my_points.npz --output my_mocks.npz

    python generate_cluster_mocks.py --list-params      # parameters, fiducials and priors

    # faster options (both off by default):
    #   --cov-fiducial   : noise covariances computed once at the fiducial point, not at each point
    #                      (~2 s instead of ~4 s per point)
    #   --no-miscentring : all the clusters centred, misc_* parameters fixed and ignored (~1 s less per
    #                      point; g_t changes by up to ~80%, and B(R) was calibrated with miscentring)
    python generate_cluster_mocks.py --n-points 10000 --ncpu 16 --cov-fiducial --output mocks_fast.npz

Output (.npz)
-------------
    theta          (n_points, n_params)        parameter values (all the parameters, fixed ones too)
    param_names    (n_params,)
    free           (n_params,)                 True for the parameters varied by --mode prior
    nc_mean        (n_points, n_z, n_l)        noise-free number counts
    gt_mean        (n_points, n_z, n_l, n_R)   noise-free reduced shear
    nc             (n_points, n_real, n_z, n_l)        noisy number counts
    gt             (n_points, n_real, n_z, n_l, n_R)   noisy reduced shear
    seed           (n_points,)                 seed of the noise of each point (reproducible)
    ok             (n_points,)                 False if the model failed at that point (NaN outputs)
    cov_nc         (n_points, n_z n_l, n_z n_l)       (with --save-cov) NC covariance, index z * n_l + l
    cov_gt         (n_points, n_z, n_l n_R, n_l n_R)  (with --save-cov) g_t covariance of each z bin,
                                                       index l * n_R + r
    nc_fid, cov_nc_fid, cov_gt_fid                    (with --cov-fiducial) fiducial counts and the
                                                       covariances used for all the points
    z_obs_edges, lambda_obs_edges, r_edges_comoving, area, and the run settings.
The output is written every --checkpoint points, and a run can be continued with --resume.

Requirements
------------
cloelib at tag sbi-mocks-v1 (``pip install -e .`` in the repository), numpy, scipy, jax,
cosmopower_jax, camb, astropy. The calibration file ``calibration_A1900_apLtr_perc.npz`` must be in the
directory of this script (or given with --calibration).
Cost: ~4 s per point and process on an idle CPU core (model ~2 s, half of it the miscentred profiles;
covariances ~2 s, mostly the g_t one), plus ~10 s of start-up per process (emulators, JAX).
"""

import os

# One thread per process: the parallelisation is done over the parameter points.
# These variables must be set before importing numpy / jax.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse
import functools
import multiprocessing as mp
import sys
import time
import warnings

import numpy as np
from scipy.integrate import simpson

import cloelib.cosmology.cosmopower_jax_cosmology as cpj
from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.emulated_perturbations import EmulatorPerturbations, NonLinearBoostPerturbations
from cloelib.observables.clusters.halo_abundance import CastroHaloAbundance
from cloelib.observables.clusters.halo_mass_observable import ShiftedPoissonHaloMassObservable
from cloelib.observables.clusters.halo_profile import BMOHaloProfile, MiscBMOHaloProfileEmu
from cloelib.observables.clusters.halo_profile.misc_bmo_halo_profile import MiscBMOHaloProfile
from cloelib.observables.clusters.matter_statistics import MatterStatistics
from cloelib.observables.clusters.selection_function.analytic_selection_function import AnalyticSelectionFunction
from cloelib.summary_statistics.clusters import ClusterCounts, ClusterStatisticsModeling, ClusterWeakLensing
from cloelib.summary_statistics.clusters.lensing import ClusterWeakLensingCovariance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION = os.path.join(SCRIPT_DIR, "calibration_A1900_apLtr_perc.npz")

########################################################################################################
# 1. Survey and binning
########################################################################################################

AREA = 1900.0  # survey area [deg2] (Euclid DR1-like); can be changed with --area
# The bins (observed redshift, observed richness, comoving radius [Mpc/h]) are read from the
# calibration file, since the PCA template of the optical selection bias is tabulated in those
# radial bins: z_ob 0.2-0.4-0.6-0.8-1.0, lambda_ob 40-60-90-150-250, 9 log bins 0.25-9 cMpc/h.

########################################################################################################
# 2. Fixed cosmological parameters (PICCOLO simulation)
########################################################################################################

H0 = 67.32  # km/s/Mpc
OMEGA_B = 0.0494
NS = 0.9661
MNU = 0.06  # eV, one massive neutrino
OMEGA_NU = MNU / 93.14 / (H0 / 100.0) ** 2

########################################################################################################
# 3. Free parameters: fiducial values ("A1900" setup) and priors
########################################################################################################
# Each entry: name -> (fiducial, prior), with prior ("uniform", min, max) or ("gauss", mean, sigma).
# Cosmology and HOD: true values of the PICCOLO mock, uniform priors of the A1900 chains.
# log10_M1 is log10 of M_1 [Msun/h].

PARAMETERS = {
    # cosmology
    "Omega_m": (0.3158, ("uniform", 0.2, 0.45)),
    "sigma8": (0.8102, ("uniform", 0.6, 1.0)),  # of the cdm + baryon linear spectrum at z = 0
    # shifted-Poisson richness-mass relation (HOD)
    "log10_M1": (12.191, ("uniform", 11.7, 12.7)),
    "alpha": (0.879, ("uniform", 0.6, 1.2)),
    "epsilon": (0.955, ("uniform", -1.0, 3.0)),
    "sigma_lnltr": (0.208, ("uniform", 0.02, 0.6)),
    # miscentring (2-Rayleigh, comoving Mpc/h): fiducials from the calibration file (filled in by
    # load_calibration); Gaussian priors of width 0.08 on f and 20% on the scales, truncated to
    # 0 < f < 1, 0 < s_1 < s_2
    "misc_f": (None, ("gauss", None, 0.08)),
    "misc_s1": (None, ("gauss", None, "20%")),
    "misc_s2": (None, ("gauss", None, "20%")),
    # systematic uncertainty of the B(R) calibration, shared by all the (z, lambda) bins (see the module
    # docstring), as the B_SYS chains: Gaussian priors of mean 0, width 0.25 on the scale B_s and
    # 0.25 x (rms over the bins of the calibrated amplitudes a_k) on the amplitude shifts B_da<k>
    # (the latter filled in by load_calibration)
    "B_s": (0.0, ("gauss", 0.0, 0.25)),
    "B_da1": (0.0, ("gauss", 0.0, None)),
    "B_da2": (0.0, ("gauss", 0.0, None)),
}
# fixed parameters of the richness-mass relation
HOD_FIXED = dict(Mmin=10.0**11.073, z_piv=1.25)

PARAMETER_GROUPS = {
    "cosmo": ["Omega_m", "sigma8"],
    "hod": ["log10_M1", "alpha", "epsilon", "sigma_lnltr"],
    "misc": ["misc_f", "misc_s1", "misc_s2"],
    "bsys": ["B_s", "B_da1", "B_da2"],
}

########################################################################################################
# 4. Numerical settings (as the A1900 chains)
########################################################################################################

SETTINGS = {
    "zed_interp": np.linspace(0.0, 2.0, 81),  # redshifts of the P(k) tables
    "ms_z": np.linspace(1e-5, 2.0 - 1e-5, 100),  # MatterStatistics tables
    "ms_k": np.geomspace(1e-4, 50.0, 800),
    "k": np.geomspace(1.1e-4, 49.0, 400),  # integration grids
    "Mass": np.geomspace(1.0e13, 1.0e16, 61),  # virial mass [Msun/h]
    "Lambda": np.geomspace(5.0, 1000.0, 160),  # true richness
    # true redshift grid of the counts: includes the clusters with z_true > 1 scattered into z_ob < 1
    "ztrue_nc": np.arange(0.01, 1.3, 0.02),
    # true redshift grid of the lensing: z_true < 1 as in the A1900 model (the profiles of the
    # PICCOLO mock, on which B(R) was calibrated, exist only for z_true < 1)
    "ztrue_gt": np.arange(0.01, 1.0, 0.02),
    "z_tab_integ": 31,  # z points of the sample variance integrals
    "n_lambda_obs": 40,  # quadrature points in each observed richness bin of the selection function
}

# The 2-halo term of the lensing model follows the PICCOLO mock, in which only the halos with
# M_vir > 1e13 Msun/h are painted (with their BMO profile out to 30 cMpc/h): halo bias b -> b W(z),
# W = int_{M > 1e13} n(M) b(M) M_paint(M) dM / rho_m, and the excluded background of the 2-halo
# term is the painted mass fraction. The PCA template of B(R) was calibrated against this model.
# PICCOLO_2HALO = False gives the standard cloelib 2-halo term (W = 1, background = 1).
PICCOLO_2HALO = True
M_MIN_PAINT = 1.0e13  # Msun/h
R_PAINT, TAU_PAINT = 30.0, 3.0  # painting radius [cMpc/h], BMO truncation [r_vir] of the mock
RHO_CRIT_H2 = 2.7751428946e11  # h^2 Msun / Mpc^3

# Lensing covariance: source density and shape noise (cloelib defaults: n_eff per tomographic bin)
N_EFF = 15.0  # gal / arcmin2 (can be changed with --n-eff)
SIGMA_E = 0.3  # shape noise per component (can be changed with --sigma-e)


########################################################################################################
# Calibration file and parameter bookkeeping
########################################################################################################


class Calibration:
    """Contents of the calibration file (selection function, miscentring, PCA template of B(R))."""

    def __init__(self, path):
        d = np.load(path)
        self.z_obs_edges = d["z_obs_edges"]
        self.lambda_obs_edges = d["lambda_obs_edges"]
        self.r_edges = d["r_edges_comoving"]  # comoving Mpc/h
        self.sel_names = [str(x) for x in d["sel_names"]]  # AnalyticSelectionFunction parameter names
        self.sel_p = d["sel_p"]  # best fit to PICCOLO
        self.misc = d["misc_comoving"]  # f, s_1, s_2 (comoving Mpc/h)
        self.pca_Bbar = d["pca_Bbar"]  # (n_R,)
        self.pca_modes = d["pca_modes"]  # (2, n_R)
        self.pca_amp = d["pca_amp"]  # (n_z, n_l, 2): calibrated amplitudes
        self.pca_amp_cov = d["pca_amp_cov"]  # (n_z, n_l, 2, 2): their covariance
        self.n_z, self.n_l = self.z_obs_edges.size - 1, self.lambda_obs_edges.size - 1
        self.n_r = self.r_edges.size - 1
        self.n_pca = self.pca_modes.shape[0]


def load_calibration(path):
    """Read the calibration file and complete PARAMETERS / PARAMETER_GROUPS with its values."""
    cal = Calibration(path)
    for name, value in zip(("misc_f", "misc_s1", "misc_s2"), cal.misc):
        _, (kind, _, sig) = PARAMETERS[name]
        if isinstance(sig, str):  # relative width, e.g. "20%"
            sig = float(sig.rstrip("%")) / 100.0 * value
        PARAMETERS[name] = (float(value), (kind, float(value), sig))
    if cal.n_pca != 2:
        raise ValueError(f"the calibration has {cal.n_pca} PCA modes, the script expects 2 (B_da1, B_da2)")
    rms = np.sqrt(np.mean(cal.pca_amp**2, axis=(0, 1)))  # rms of the calibrated amplitudes over the bins
    for k in range(cal.n_pca):
        PARAMETERS[f"B_da{k + 1}"] = (0.0, ("gauss", 0.0, 0.25 * float(rms[k])))
    return cal


def expand_fixed(spec):
    """'--fix cosmo,misc_f' -> set of parameter names"""
    out = set()
    for item in filter(None, (s.strip() for s in spec.split(","))):
        if item in PARAMETER_GROUPS:
            out.update(PARAMETER_GROUPS[item])
        elif item in PARAMETERS:
            out.add(item)
        else:
            raise ValueError(f"--fix: unknown parameter or group '{item}'")
    return out


def sample_prior(n, rng, cal, fixed=()):
    """n points from the prior, fixed parameters at their fiducial: (n, n_params)"""
    names = list(PARAMETERS)
    theta = np.tile([PARAMETERS[p][0] for p in names], (n, 1))
    col = {p: names.index(p) for p in names}
    for p in names:
        if p in fixed:
            continue
        prior = PARAMETERS[p][1]
        if prior[0] == "uniform":
            theta[:, col[p]] = rng.uniform(prior[1], prior[2], n)
        elif p in PARAMETER_GROUPS["bsys"]:  # Gaussian, not truncated
            theta[:, col[p]] = rng.normal(prior[1], prior[2], n)
    # miscentring: Gaussian, truncated to 0 < f < 1 and 0 < s_1 < s_2 (rejection sampling)
    free_misc = [p for p in PARAMETER_GROUPS["misc"] if p not in fixed]
    todo = np.ones(n, bool)
    while free_misc and todo.any():
        m = int(todo.sum())
        for p in free_misc:
            _, (_, mu, sig) = PARAMETERS[p]
            theta[todo, col[p]] = rng.normal(mu, sig, m)
        f, s1, s2 = (theta[:, col[p]] for p in PARAMETER_GROUPS["misc"])
        todo = ~((f > 0) & (f < 1) & (s1 > 0) & (s1 < s2))
    return theta


def print_parameters(fixed=()):
    print(f"{'parameter':14s} {'fiducial':>10s}  prior")
    for p, (fid, prior) in PARAMETERS.items():
        if prior[0] == "uniform":
            txt = f"U({prior[1]:g}, {prior[2]:g})"
        else:
            txt = f"N({prior[1]:.4g}, {prior[2]:.3g})"
            if p in PARAMETER_GROUPS["misc"]:
                txt += " truncated (0 < f < 1, 0 < s1 < s2)"
        print(f"{p:14s} {fid:10.4f}  {txt}{'  [fixed]' if p in fixed else ''}")


########################################################################################################
# 5. Model
########################################################################################################

# The CosmoPower-JAX emulators of cloelib are read from disk at every construction of their classes:
# cache them, since a new cosmology is built at every parameter point.
cpj.load_pk_emulator = functools.lru_cache(maxsize=None)(cpj.load_pk_emulator)
cpj.load_sigma_emulator = functools.lru_cache(maxsize=None)(cpj.load_sigma_emulator)


def _background(Omega_m, As):
    """flat LCDM background with one massive neutrino"""
    return CAMBBackground(
        H0=H0, Omega_cdm0=Omega_m - OMEGA_B - OMEGA_NU, Omega_b0=OMEGA_B, Omega_k0=0.0, w0=-1.0, wa=0.0,
        ns=NS, mnu=MNU, As=As, gamma_MG=0.0, N_mnu=1,
    )


def _linear_perturbations(background, redshifts):
    """linear total matter and cdm + baryon P(k) from the cloelib CosmoPower-JAX emulators"""
    emu = cpj.CosmoPowerJAXLCDMPerturbations
    return EmulatorPerturbations(
        background, emu.Linear(background, redshifts), emu.LinearCB(background, redshifts)
    )


_K8 = np.geomspace(1e-4, 20.0, 4000)


def _sigma8_cb(perturbations):
    """sigma8 of the cdm + baryon linear spectrum at z = 0 (top hat of 8 Mpc/h)"""
    x = 8.0 * _K8
    w = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    pk = perturbations.matter_power_spectrum_cb(np.array([0.0]), _K8, hubble_units=True, k_hunit=True)[0]
    return np.sqrt(simpson(_K8**3 * pk * w**2 / (2.0 * np.pi**2), x=np.log(_K8)))


def get_perturbations(Omega_m, sigma8):
    """linear perturbations with the requested sigma8 (cdm + baryons), set by rescaling A_s"""
    As_ref = 2.1e-9
    sigma8_ref = _sigma8_cb(_linear_perturbations(_background(Omega_m, As_ref), np.array([0.0, 0.1])))
    return _linear_perturbations(_background(Omega_m, As_ref * (sigma8 / sigma8_ref) ** 2), SETTINGS["zed_interp"])


def c_of_m(z, M, A=7.85, B=-0.081, C=-0.71):
    """Duffy et al. 2008 concentration of M_vir: (z.size, M.size)"""
    return A * (np.atleast_1d(M)[None, :] / 2.0e12) ** B * (1.0 + np.atleast_1d(z)[:, None]) ** C


def m_painted_ratio(z, M, Omega_m):
    """M_paint / M_vir, (z.size, M.size): projected mass of the BMO profile (truncation TAU_PAINT r_vir,
    Duffy c) within the painting radius R_PAINT of the PICCOLO mock (only used with PICCOLO_2HALO)"""
    zc = np.atleast_1d(z)[:, None]
    E2 = Omega_m * (1.0 + zc) ** 3 + 1.0 - Omega_m
    y = Omega_m * (1.0 + zc) ** 3 / E2 - 1.0
    r_vir = (3.0 * np.atleast_1d(M)[None, :] / (4.0 * np.pi * (18.0 * np.pi**2 + 82.0 * y - 39.0 * y**2)
                                                 * RHO_CRIT_H2 * E2)) ** (1.0 / 3.0)  # pMpc/h
    c = c_of_m(z, M)
    t = TAU_PAINT * c
    # M_vir / (4 pi rho_s r_s^3) (Eq. 2 of arXiv:1101.0650)
    f_c = t**2 / 2.0 / (t**2 + 1.0) ** 3 / (1.0 + c) / (t**2 + c**2) * (
        c * (t**2 + 1.0) * (c * (c + 1) - t**2 * (c - 1.0) * (2.0 + 3 * c) - 2.0 * t**4)
        + t * (c + 1) * (t**2 + c**2) * (2.0 * (3.0 * t**2 - 1) * np.arctan(c / t)
                                         + t * (t**2 - 3.0) * np.log(t**2 * (1.0 + c) ** 2 / (t**2 + c**2)))
    )
    # projected mass within R_PAINT / (4 pi rho_s r_s^3) (Eq. A.29 of arXiv:0705.0682)
    x = R_PAINT / (1.0 + zc) / (r_vir / c)
    F = np.arccos(1.0 / x) / np.sqrt(x**2 - 1.0)
    L = np.log(x / (np.sqrt(t**2 + x**2) + t))
    m_prj = (t**4 / 2.0 / (t**2 + 1.0) ** 3) * (
        2.0 * (t**2 + 1.0 + 4.0 * (x**2 - 1.0)) * F
        + 1.0 / t * (np.pi * (3.0 * t**2 - 1.0) + 2.0 * t * (t**2 - 3.0) * np.log(t))
        + 1.0 / t**3 / np.sqrt(t**2 + x**2) * (
            -(t**3) * np.pi * (4.0 * (t**2 + x**2) - t**2 - 1.0)
            + (-(t**2) * (t**4 - 1.0) + (t**2 + x**2) * (3.0 * t**4 - 6.0 * t**2 - 1.0)) * L
        )
    )
    return m_prj / f_c


def painted_fraction(castro, z, with_bias, Omega_m, M_max=1.0e16, n=400):
    """int_{M > M_MIN_PAINT} n(M) [b(M)] M_paint(M) dM / rho_m, (z.size,): W(z) (with_bias) or the
    painted mass fraction (without)"""
    bg = castro.core.matter_statistics.background
    rho_m = bg.Omega_m(0.0) * derived_cosmology.rho_crit(bg, 0.0) / bg.h**2
    M = np.geomspace(M_MIN_PAINT, M_max, n)
    integ = castro.dn_dm(z, M) * m_painted_ratio(z, M, Omega_m) * M[None, :] ** 2
    if with_bias:
        integ = integ * castro.bias(z, M)
    return np.trapezoid(integ, np.log(M), axis=1) / rho_m


class ComovingOffsetProfile:
    """Miscentred profile whose offset scale sigma_off is given in comoving Mpc/h: the wrapped cloelib
    profile receives sigma_off / (1 + z). ClusterWeakLensing.get_gt with comoving_radii=True evaluates
    the profiles at one true redshift per call, which this wrapper requires."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _physical(self, z, extra):
        if np.size(z) != 1:
            raise ValueError("ComovingOffsetProfile needs one redshift per call")
        return (extra[0] / (1.0 + float(np.ravel(z)[0])),) + tuple(extra[1:])

    def excess_surface_mass_density(self, R, z, M, c, *extra, **kw):
        return self.base.excess_surface_mass_density(R, z, M, c, *self._physical(z, extra), **kw)

    def surface_mass_density(self, R, z, M, c, *extra, **kw):
        return self.base.surface_mass_density(R, z, M, c, *self._physical(z, extra), **kw)


class ClusterModel:
    """Mean observables and covariances at a parameter point."""

    def __init__(self, cal, area=AREA, n_eff=N_EFF, sigma_e=SIGMA_E, misc_profile="emu", miscentring=True):
        """miscentring=False: all the clusters are centred (the misc_* parameters are ignored), which
        halves the cost of the lensing model; note that B(R) was calibrated with miscentring."""
        self.cal, self.area, self.n_eff, self.sigma_e = cal, area, n_eff, sigma_e
        self.misc_profile, self.miscentring = misc_profile, miscentring
        self.names = list(PARAMETERS)

    def _modeling(self, castro, ztrue):
        return ClusterStatisticsModeling(
            castro, integ_k_arr=SETTINGS["k"], integ_mass_arr=SETTINGS["Mass"],
            integ_lambda_true_arr=SETTINGS["Lambda"], integ_ztrue_arr=ztrue, area=self.area,
        )

    def __call__(self, theta, with_covariance=True):
        """theta: values of all the parameters (order of PARAMETERS).
        Returns nc (n_z, n_l), gt (n_z, n_l, n_R) and, with_covariance, cov_nc (n_z n_l, n_z n_l) and
        cov_gt (n_z, n_l n_R, n_l n_R)."""
        cal = self.cal
        p = dict(zip(self.names, theta))
        zob, lob, re = cal.z_obs_edges, cal.lambda_obs_edges, cal.r_edges

        # --- cosmology: linear P(k), nonlinear P_cb (2-halo term with exclusion), HMF and bias
        perturbations = get_perturbations(p["Omega_m"], p["sigma8"])
        nonlinear = NonLinearBoostPerturbations(perturbations, SETTINGS["zed_interp"])
        matter_stat = MatterStatistics(
            perturbations, z=SETTINGS["ms_z"], k=SETTINGS["ms_k"], nonlinear_perturbations=nonlinear
        )
        castro = CastroHaloAbundance(matter_stat)

        # --- richness-mass relation and selection function
        hmo = ShiftedPoissonHaloMassObservable(
            Mmin=HOD_FIXED["Mmin"], M1=10.0 ** p["log10_M1"], alpha=p["alpha"], epsilon=p["epsilon"],
            sigma_lnltr=p["sigma_lnltr"], z_piv=HOD_FIXED["z_piv"],
        )
        selection = AnalyticSelectionFunction(
            hmo, **dict(zip(cal.sel_names, cal.sel_p)),
            z_tab_integ=SETTINGS["z_tab_integ"], lambda_tab_integ=[SETTINGS["n_lambda_obs"]] * cal.n_l,
        )

        # --- number counts (and their covariance: shot noise + sample variance)
        csm_nc = self._modeling(castro, SETTINGS["ztrue_nc"])
        halo_cov = None
        if with_covariance:
            halo_cov = HaloCovariance(
                perturbations, k=SETTINGS["k"], area=self.area, nbins_zob=cal.n_z, z_tab_integ=SETTINGS["z_tab_integ"]
            )
        counts = ClusterCounts(csm_nc, halo_cov, selection)
        nc, aux = counts.get_NC(zob, lob, return_intermediate_products=True)

        # --- reduced shear profiles
        csm_gt = self._modeling(castro, SETTINGS["ztrue_gt"])
        bias_scale = np.ones(csm_gt.tabulated_integrands["ztrue"].size)
        exclusion_background = None
        if PICCOLO_2HALO:
            tab = csm_gt.tabulated_integrands
            bias_scale = painted_fraction(castro, tab["ztrue"], True, p["Omega_m"])
            tab["bias(ztrue,M)"] = tab["bias(ztrue,M)"] * bias_scale[:, None]
            z_f = np.linspace(0.0, 2.0, 41)
            f_painted = painted_fraction(castro, z_f, False, p["Omega_m"])
            exclusion_background = lambda z: np.interp(z, z_f, f_painted)  # noqa: E731
        profile_kwargs = dict(two_halo="sum_exclusion", trunc_fact=3.0, zs_max=2.0)
        if exclusion_background is not None:
            profile_kwargs["exclusion_background"] = exclusion_background
        centred = BMOHaloProfile(matter_stat, **profile_kwargs)
        miscentred, mixture = None, None  # no miscentring: all the clusters centred
        if self.miscentring:
            misc_cls = MiscBMOHaloProfileEmu if self.misc_profile == "emu" else MiscBMOHaloProfile
            miscentred = ComovingOffsetProfile(misc_cls(matter_stat, **profile_kwargs))
            f, s1, s2 = p["misc_f"], p["misc_s1"], p["misc_s2"]
            mixture = ((f, s1), (1.0 - f, s2))
        weak_lensing = ClusterWeakLensing(
            csm_gt, centred, halo_concentration=c_of_m, selection_function=selection, miscentered_profile=miscentred
        )
        # optical selection bias, with the systematic parameters B_s, B_da<k>:
        #     B - 1 = (1 + B_s) [Bbar - 1 + sum_k (a_k + B_da<k>) e_k]
        # (a_k: calibrated amplitudes of each bin, fixed)
        s_b = p["B_s"]
        da = np.array([p[f"B_da{k + 1}"] for k in range(cal.n_pca)])
        b_mean = 1.0 + (1.0 + s_b) * (cal.pca_Bbar - 1.0)
        amp = (1.0 + s_b) * (cal.pca_amp + da)  # (n_z, n_l, n_pca)
        # profile without the optical selection bias correction, then times B(R) (same as passing
        # opt_sel_bias_pca=(b_mean, modes, amp) to get_gt; the uncorrected profile is also needed for
        # the PCA covariance below)
        gt_no_b = weak_lensing.get_gt(zob, lob, re, miscentering_params=mixture, comoving_radii=True)
        gt = gt_no_b * weak_lensing.optical_selection_bias_pca_correction(b_mean, cal.pca_modes, amp)
        if not with_covariance:
            return nc, gt

        # --- covariances
        # NC: (z, z, l, l) -> (z l, z l)
        cov_nc = counts.get_NC_covariance(zob, nc, aux["window_z_lambda_obs"])
        n_nc = cal.n_z * cal.n_l
        cov_nc = cov_nc.transpose(0, 2, 1, 3).reshape(n_nc, n_nc)
        # g_t: the cloelib lensing covariance takes physical radii; the bins are comoving, so it is
        # computed for each z_ob bin with R_phys = R_com / (1 + z_mid). Its lens sample uses the halo
        # bias of the counts (the 2-halo rescaling of the mock does not apply to the halo clustering).
        weak_lensing_cov = ClusterWeakLensing(
            csm_nc, centred, halo_concentration=c_of_m, selection_function=selection, miscentered_profile=miscentred
        )
        cov_obj = ClusterWeakLensingCovariance(
            weak_lensing_cov, n_eff=self.n_eff, sigma_e=self.sigma_e, nonlinear_perturbations=nonlinear
        )
        n_lr = cal.n_l * cal.n_r
        cov_gt = np.zeros((cal.n_z, n_lr, n_lr))
        for i in range(cal.n_z):
            z_mid = 0.5 * (zob[i] + zob[i + 1])
            c = cov_obj.get_gt_covariance(zob[i : i + 2], lob, re / (1.0 + z_mid))[0, 0]  # (l, l, r, r)
            cov_gt[i] = c.transpose(0, 2, 1, 3).reshape(n_lr, n_lr)
        # g_t: statistical uncertainty of the calibrated amplitudes a_k (fixed in the model), propagated
        # with the profile without B, added to each (z, lambda) block (as the A1900 and B_SYS chains)
        c = weak_lensing.optical_selection_bias_pca_covariance(gt_no_b, cal.pca_modes, cal.pca_amp_cov)
        for i in range(cal.n_z):
            for j in range(cal.n_l):
                s = slice(j * cal.n_r, (j + 1) * cal.n_r)
                cov_gt[i, s, s] += c[i, i, j, j]
        return nc, gt, cov_nc, cov_gt


########################################################################################################
# 6. Noise
########################################################################################################


def _cholesky(cov):
    """Cholesky factor, with a small diagonal jitter if the matrix is not numerically positive definite"""
    jitter = 0.0
    scale = np.mean(np.diag(cov))
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + jitter * scale * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0.0 else 10.0 * jitter
    raise np.linalg.LinAlgError("covariance not positive definite")


def add_noise(nc, gt, cov_nc, cov_gt, n_real, rng, nc_noise="gaussian", nc_cov=None):
    """n_real noisy realisations of (nc, gt).
    nc_noise = "gaussian": N ~ Gauss(nc, cov_nc), negative counts set to 0 (rounded to integers);
    nc_noise = "poisson" : sample variance as a Gaussian field (cov_nc minus its shot-noise diagonal),
                           then Poisson counts of that mean (clipped at 0): integer counts, Poisson
                           statistics in the poorly populated bins.
    nc_cov: the counts at which cov_nc was computed (its shot noise is diag(nc_cov)), if not nc
    (covariance fixed at the fiducial point, --cov-fiducial)."""
    n_z, n_l, n_r = gt.shape
    if nc_noise == "gaussian":
        L = _cholesky(cov_nc)
        nc_r = nc.ravel()[None, :] + rng.standard_normal((n_real, nc.size)) @ L.T
        nc_r = np.rint(np.clip(nc_r, 0.0, None))
    else:
        # sample variance part (shot noise = diagonal of N)
        cov_sv = cov_nc - np.diag((nc if nc_cov is None else nc_cov).ravel())
        L = _cholesky(cov_sv)
        mean = np.clip(nc.ravel()[None, :] + rng.standard_normal((n_real, nc.size)) @ L.T, 0.0, None)
        nc_r = rng.poisson(mean).astype(float)
    gt_r = np.empty((n_real, n_z, n_l, n_r))
    for i in range(n_z):
        L = _cholesky(cov_gt[i])
        gt_r[:, i] = (gt[i].ravel()[None, :] + rng.standard_normal((n_real, n_l * n_r)) @ L.T).reshape(n_real, n_l, n_r)
    return nc_r.reshape(n_real, n_z, n_l), gt_r


########################################################################################################
# 7. Driver (parallel over the parameter points)
########################################################################################################

_WORKER = {}


def _init_worker(cfg):
    """per-process setup: calibration and model (the emulators are loaded at the first call)"""
    warnings.filterwarnings("ignore")
    cal = load_calibration(cfg["calibration"])
    _WORKER["model"] = ClusterModel(cal, area=cfg["area"], n_eff=cfg["n_eff"], sigma_e=cfg["sigma_e"],
                                    misc_profile=cfg["misc_profile"], miscentring=cfg["miscentring"])
    _WORKER["cfg"] = cfg


def _run_point(task):
    """model + noise at one point: task = (index, theta, seed)"""
    index, theta, seed = task
    cfg, model = _WORKER["cfg"], _WORKER["model"]
    cal = model.cal
    t0 = time.time()
    try:
        rng = np.random.default_rng(seed)
        if cfg["cov_fiducial"] is None:  # covariances at this point
            nc, gt, cov_nc, cov_gt = model(theta, with_covariance=True)
            nc_r, gt_r = add_noise(nc, gt, cov_nc, cov_gt, cfg["n_real"], rng, cfg["nc_noise"])
        else:  # covariances computed once at the fiducial point (--cov-fiducial)
            nc, gt = model(theta, with_covariance=False)
            nc_fid, cov_nc, cov_gt = cfg["cov_fiducial"]
            nc_r, gt_r = add_noise(nc, gt, cov_nc, cov_gt, cfg["n_real"], rng, cfg["nc_noise"], nc_cov=nc_fid)
            cov_nc = cov_gt = None  # saved once in the output, not per point
        ok = bool(np.all(np.isfinite(nc)) and np.all(np.isfinite(gt)))
    except Exception as err:  # failed point: NaN outputs, flagged in "ok"
        print(f"  point {index}: model failed ({type(err).__name__}: {err})", flush=True)
        nan = np.nan
        nc = np.full((cal.n_z, cal.n_l), nan)
        gt = np.full((cal.n_z, cal.n_l, cal.n_r), nan)
        cov_nc = np.full((nc.size, nc.size), nan)
        cov_gt = np.full((cal.n_z, cal.n_l * cal.n_r, cal.n_l * cal.n_r), nan)
        nc_r = np.full((cfg["n_real"],) + nc.shape, nan)
        gt_r = np.full((cfg["n_real"],) + gt.shape, nan)
        ok = False
    return index, nc, gt, nc_r, gt_r, cov_nc, cov_gt, ok, time.time() - t0


def _save(path, results, base):
    """write the points computed so far (sorted by index)"""
    idx = np.array(sorted(results))
    out = dict(base)
    out["theta"] = base["theta"][idx]
    out["seed"] = base["seed"][idx]
    out["index"] = idx
    keys = ("nc_mean", "gt_mean", "nc", "gt", "cov_nc", "cov_gt", "ok")
    for k, pos in zip(keys, range(1, 8)):
        if k.startswith("cov") and not base["save_cov"]:
            continue
        out[k] = np.array([results[i][pos] for i in idx])
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)


def _load_theta_file(path):
    names = list(PARAMETERS)
    if path.endswith(".npz"):
        d = np.load(path)
        n = len(d[d.files[0]])
        theta = np.tile([PARAMETERS[p][0] for p in names], (n, 1))
        for k in d.files:
            if k not in PARAMETERS:
                raise ValueError(f"{path}: unknown parameter '{k}'")
            theta[:, names.index(k)] = d[k]
        return theta
    theta = np.atleast_2d(np.load(path))
    if theta.shape[1] != len(names):
        raise ValueError(f"{path}: {theta.shape[1]} columns, {len(names)} parameters (see --list-params)")
    return theta


def main():
    parser = argparse.ArgumentParser(
        description="Mock cluster number counts and reduced shear profiles (cloelib, A1900 setup)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("prior", "fiducial", "file"), default="prior",
                        help="parameter points: drawn from the prior, the fiducial point, or read from --theta-file")
    parser.add_argument("--n-points", type=int, default=10, help="number of points (mode prior)")
    parser.add_argument("--n-real", type=int, default=1, help="noise realisations per point")
    parser.add_argument("--theta-file", help="parameter points (mode file): .npy (n, n_params) or .npz (one array per name)")
    parser.add_argument("--fix", default="", help="parameters or groups (cosmo, hod, misc, bsys) fixed to the fiducial")
    parser.add_argument("--cov-fiducial", action="store_true",
                        help="noise covariances computed once at the fiducial point instead of at each point "
                             "(~2x faster); saved in the output as cov_nc_fid, cov_gt_fid")
    parser.add_argument("--no-miscentring", action="store_true",
                        help="all the clusters centred (misc_* parameters fixed and ignored; ~1 s faster per point). "
                             "Note: B(R) was calibrated with miscentring")
    parser.add_argument("--nc-noise", choices=("gaussian", "poisson"), default="gaussian",
                        help="NC noise: Gaussian with the full covariance (clipped at 0), or Poisson x Gaussian sample variance")
    parser.add_argument("--area", type=float, default=AREA, help="survey area [deg2]")
    parser.add_argument("--n-eff", type=float, default=N_EFF, help="source density [gal/arcmin2] per tomographic bin")
    parser.add_argument("--sigma-e", type=float, default=SIGMA_E, help="shape noise per component")
    parser.add_argument("--misc-profile", choices=("emu", "numerical"), default="emu",
                        help="miscentred profile: cloelib emulator (fast) or numerical integration (exact, slower)")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION, help="calibration file")
    parser.add_argument("--seed", type=int, default=12345, help="master seed (parameter draws and noise)")
    parser.add_argument("--ncpu", type=int, default=1, help="number of processes")
    parser.add_argument("--save-cov", action="store_true", help="also save the covariances of each point")
    parser.add_argument("--checkpoint", type=int, default=50, help="write the output every this many points")
    parser.add_argument("--resume", action="store_true", help="continue a run with the same output file and options")
    parser.add_argument("--output", default="cluster_mocks.npz", help="output file (.npz)")
    parser.add_argument("--list-params", action="store_true", help="print the parameters and exit")
    args = parser.parse_args()

    cal = load_calibration(args.calibration)
    fixed = expand_fixed(args.fix)
    if args.no_miscentring:  # the miscentring parameters do not enter the model
        fixed |= set(PARAMETER_GROUPS["misc"])
    if args.list_params:
        print_parameters(fixed)
        return
    names = list(PARAMETERS)

    # parameter points and noise seeds: point i is drawn with its own seed, derived from the master
    # seed and i only, so a run is reproducible and can be resumed or extended (larger --n-points)
    seq = np.random.SeedSequence(args.seed)
    seq_theta, seq_noise = seq.spawn(2)
    if args.mode == "prior":
        theta = np.concatenate(
            [sample_prior(1, np.random.default_rng(s), cal, fixed) for s in seq_theta.spawn(args.n_points)]
        )
    elif args.mode == "fiducial":
        theta = np.array([[PARAMETERS[p][0] for p in names]])
    else:
        if not args.theta_file:
            parser.error("--mode file needs --theta-file")
        theta = _load_theta_file(args.theta_file)
    n_points = theta.shape[0]
    seeds = np.array([s.generate_state(1)[0] for s in seq_noise.spawn(n_points)], dtype=np.uint32)
    free =np.array([args.mode == "prior" and p not in fixed for p in names])

    base = dict(
        param_names=np.array(names), free=free, theta=theta, seed=seeds,
        fiducial=np.array([PARAMETERS[p][0] for p in names]),
        z_obs_edges=cal.z_obs_edges, lambda_obs_edges=cal.lambda_obs_edges, r_edges_comoving=cal.r_edges,
        area=args.area, n_eff=args.n_eff, sigma_e=args.sigma_e, nc_noise=args.nc_noise, mode=args.mode,
        misc_profile=args.misc_profile, master_seed=args.seed, miscentring=not args.no_miscentring,
        cov_fiducial=args.cov_fiducial, piccolo_2halo=PICCOLO_2HALO,
        # per-point covariances only if they are computed at each point
        save_cov=args.save_cov and not args.cov_fiducial, command=" ".join(sys.argv),
    )

    # --cov-fiducial: covariances computed once, here, at the fiducial point (with the same model options)
    cov_fiducial = None
    if args.cov_fiducial:
        t0 = time.time()
        warnings.filterwarnings("ignore")
        model_fid = ClusterModel(cal, area=args.area, n_eff=args.n_eff, sigma_e=args.sigma_e,
                                 misc_profile=args.misc_profile, miscentring=not args.no_miscentring)
        nc_fid, _, cov_nc_fid, cov_gt_fid = model_fid(base["fiducial"], with_covariance=True)
        cov_fiducial = (nc_fid, cov_nc_fid, cov_gt_fid)
        base.update(nc_fid=nc_fid, cov_nc_fid=cov_nc_fid, cov_gt_fid=cov_gt_fid)
        print(f"Covariances at the fiducial point: {time.time() - t0:.1f} s", flush=True)

    results = {}
    if args.resume and os.path.exists(args.output):
        old = np.load(args.output)
        if not np.allclose(old["theta"], theta[old["index"]]):
            sys.exit(f"--resume: {args.output} was made with different parameter points")
        for n, i in enumerate(old["index"]):
            results[int(i)] = (int(i), old["nc_mean"][n], old["gt_mean"][n], old["nc"][n], old["gt"][n],
                               old["cov_nc"][n] if base["save_cov"] else None,
                               old["cov_gt"][n] if base["save_cov"] else None, bool(old["ok"][n]), 0.0)
        print(f"Resuming: {len(results)} of {n_points} points already in {args.output}")

    cfg = dict(calibration=args.calibration, area=args.area, n_eff=args.n_eff, sigma_e=args.sigma_e,
               misc_profile=args.misc_profile, n_real=args.n_real, nc_noise=args.nc_noise,
               miscentring=not args.no_miscentring, cov_fiducial=cov_fiducial)
    tasks = [(i, theta[i], int(seeds[i])) for i in range(n_points) if i not in results]
    print(f"{len(tasks)} points x {args.n_real} realisations, {args.ncpu} processes -> {args.output}", flush=True)

    t_start = time.time()
    done_since_save = 0
    if args.ncpu > 1:
        # "spawn": JAX does not work in forked processes
        pool = mp.get_context("spawn").Pool(args.ncpu, initializer=_init_worker, initargs=(cfg,))
        iterator = pool.imap_unordered(_run_point, tasks)
    else:
        pool = None
        _init_worker(cfg)
        iterator = map(_run_point, tasks)
    try:
        for res in iterator:
            if not base["save_cov"]:  # do not keep the covariances in memory
                res = res[:5] + (None, None) + res[7:]
            results[res[0]] = res
            done_since_save += 1
            n_done = len(results)
            print(f"  point {res[0]:6d} done in {res[-1]:6.1f} s  ({n_done}/{n_points}, "
                  f"{(time.time() - t_start) / 60:.1f} min)", flush=True)
            if done_since_save >= args.checkpoint:
                _save(args.output, results, base)
                done_since_save = 0
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if results:
            _save(args.output, results, base)
    n_bad = sum(not r[7] for r in results.values())
    print(f"Done: {len(results)} points in {(time.time() - t_start) / 60:.1f} min ({n_bad} failed) -> {args.output}")


if __name__ == "__main__":
    main()
