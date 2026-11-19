#!/usr/bin/env python
"""
Likelihood-based inference (nested sampling with dynesty) on a mock made by generate_cluster_mocks.py,
with the same cloelib model, parameters and priors: the reference posterior to compare SBI with.

Data
----
One data vector of a mock file of generate_cluster_mocks.py: point --index of the file (default 0),
noise realisation --real (default 0), or the noise-free model of that point with --noise-free. The
model settings (survey area, source density, shape noise, miscentring on/off, miscentred profile) are
read from the mock file, so that data and model are consistent. The true parameters of the point are
stored with the chain.

Likelihood
----------
Gaussian, NC and g_t independent (as the noise of the mocks):
    -2 ln L = (d_NC - m_NC)^T C_NC^-1 (d_NC - m_NC) + sum_z (d_gt,z - m_gt,z)^T C_gt,z^-1 (d_gt,z - m_gt,z)
              [+ ln det C_NC + sum_z ln det C_gt,z  with --cov varying]
with the cloelib covariances (NC: Poisson + sample variance; g_t: shape noise + LSS + intrinsic +
calibration uncertainty of the B(R) amplitudes) computed (--cov):
* fiducial (default): once, at the fiducial parameters (as the DR1 chains);
* truth            : once, at the true parameters of the mock point;
* varying          : at each likelihood call, with the ln det term (the exact likelihood of the
                     Gaussian noise of the mocks; ~2x slower per call).
The mocks made with --cov-fiducial have their noise drawn with the fiducial covariance: analyse them
with --cov fiducial. The default mocks have noise drawn with the covariance of each point: --cov truth
(or varying) is the consistent choice. The NC noise of the mocks is rounded to integers and clipped at
0 (or Poisson with --nc-noise poisson): the Gaussian likelihood is an approximation in the bins with
few clusters.

Parameters and priors
---------------------
Those of generate_cluster_mocks.py (see --list-params there): uniform priors on cosmology and HOD,
Gaussian priors on miscentring (truncated to 0 < f < 1, 0 < s_1 < s_2) and B(R) systematics.
--fix fixes parameters or groups (cosmo, hod, misc, bsys) to their TRUE value (that of the mock point).
With --observables nc the miscentring and B systematics are fixed (they only enter the lensing), as
they are when the mock was made without miscentring.

Usage
-----
    # mock: 1 noisy realisation at the fiducial point
    python generate_cluster_mocks.py --mode fiducial --n-real 1 --output mock_fid.npz
    # chain on it, 16 processes. Cost: ~1e5 likelihood calls (12 parameters, nlive 500), of ~2 s for
    # NC + g_t (a few hours on 16 cores) and ~0.25 s for NC only (~30 min on 16 cores)
    python run_chain.py --data mock_fid.npz --ncpu 16 --output chain_fid.npz
    python run_chain.py --data mock_fid.npz --observables nc --ncpu 16 --output chain_fid_nc.npz
    # point 12, realisation 3 of a prior run, covariance at each step, B systematics fixed
    python run_chain.py --data mocks_prior.npz --index 12 --real 3 --cov varying --fix bsys --output chain_12.npz

An interrupted chain is resumed from its checkpoint file (<output>.save) by running the same command.

Output (.npz)
-------------
    samples (n, n_free), weights (n,), logl (n,), logz, logzerr   dynesty results (weights sum to 1)
    param_names (n_free,), truth (n_free,), fiducial (n_free,)     sampled parameters
    all_param_names, theta_true (all the parameters of the mock point), fixed (names)
    nc_data, gt_data, cov_nc, cov_gt (the fixed covariances, not with --cov varying), and the settings.
"""

import os

# One thread per process: the parallelisation is done over the dynesty live points.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse
import multiprocessing as mp
import sys
import time
import warnings

import numpy as np
from scipy.special import ndtri

import dynesty
from dynesty import utils as dyfunc

# the model, parameters and priors of the mock generator (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_cluster_mocks as gcm  # noqa: E402

BAD_LOGL = -1.0e100  # log-likelihood outside the (truncated) prior or where the model fails

########################################################################################################
# Global state: built by setup() in the main process and in each worker of the pool
########################################################################################################

S = {}


def setup(cfg):
    """model, data, covariances and parameter bookkeeping from the run configuration cfg"""
    warnings.filterwarnings("ignore")
    gcm.load_calibration(cfg["calibration"])  # completes the parameter table (miscentring, B priors)
    S["cfg"] = cfg
    S["model"] = gcm.ClusterModel(
        gcm.Calibration(cfg["calibration"]), area=cfg["area"], n_eff=cfg["n_eff"], sigma_e=cfg["sigma_e"],
        misc_profile=cfg["misc_profile"], miscentring=cfg["miscentring"], observables=cfg["observables"],
    )
    names = list(gcm.PARAMETERS)
    S["names"] = names
    S["free"] = [p for p in names if p not in cfg["fixed"]]
    S["i_free"] = np.array([names.index(p) for p in S["free"]], dtype=int)
    S["theta_base"] = np.asarray(cfg["theta_true"], float).copy()  # fixed parameters at the true values
    # prior of each free parameter: ("uniform", min, max) or ("gauss", mean, sigma)
    S["priors"] = [gcm.PARAMETERS[p][1] for p in S["free"]]
    S["misc_idx"] = [names.index(p) for p in gcm.PARAMETER_GROUPS["misc"]]
    S["nc_data"], S["gt_data"] = cfg["nc_data"], cfg["gt_data"]
    S["cov"] = cfg["cov"]  # None (varying) or (cov_nc, cov_gt)
    if S["cov"] is not None:
        cov_nc, cov_gt = S["cov"]
        S["inv_nc"] = None if cov_nc is None else np.linalg.inv(cov_nc)
        S["inv_gt"] = None if cov_gt is None else np.linalg.inv(cov_gt)


def full_theta(x):
    """all the parameters (order of gcm.PARAMETERS) from the sampled ones"""
    theta = S["theta_base"].copy()
    theta[S["i_free"]] = x
    return theta


def prior_transform(u):
    """unit cube -> parameters (uniform or Gaussian priors)"""
    x = np.empty_like(u)
    for k, (prior, uk) in enumerate(zip(S["priors"], u)):
        if prior[0] == "uniform":
            x[k] = prior[1] + uk * (prior[2] - prior[1])
        else:
            x[k] = prior[1] + prior[2] * ndtri(uk)
    return x


def _chi2_logdet(r, cov=None, inv=None):
    """r^T C^-1 r (and ln det C if cov is given): r (n,), C (n, n)"""
    if inv is not None:
        return r @ inv @ r, 0.0
    L = np.linalg.cholesky(cov)
    y = np.linalg.solve(L, r)
    return y @ y, 2.0 * np.sum(np.log(np.diag(L)))


def loglike(x):
    theta = full_theta(x)
    f, s1, s2 = theta[S["misc_idx"]]
    if not (0.0 < f < 1.0 and 0.0 < s1 < s2):  # truncation of the miscentring prior
        return BAD_LOGL
    varying = S["cov"] is None
    try:
        if varying:
            nc, gt, cov_nc, cov_gt = S["model"](theta, with_covariance=True)
        else:
            nc, gt = S["model"](theta, with_covariance=False)
        m2lnl = 0.0
        if nc is not None:
            r = (S["nc_data"] - nc).ravel()
            c, ld = _chi2_logdet(r, cov=cov_nc) if varying else _chi2_logdet(r, inv=S["inv_nc"])
            m2lnl += c + ld
        if gt is not None:
            n_z = gt.shape[0]
            for i in range(n_z):  # independent z_ob bins
                r = (S["gt_data"][i] - gt[i]).ravel()
                c, ld = _chi2_logdet(r, cov=cov_gt[i]) if varying else _chi2_logdet(r, inv=S["inv_gt"][i])
                m2lnl += c + ld
    except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError):
        return BAD_LOGL
    return -0.5 * m2lnl if np.isfinite(m2lnl) else BAD_LOGL


########################################################################################################
# Driver
########################################################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Nested sampling (dynesty) of a mock of generate_cluster_mocks.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="mock file made by generate_cluster_mocks.py")
    parser.add_argument("--index", type=int, default=0, help="point of the mock file (row of theta)")
    parser.add_argument("--real", type=int, default=0, help="noise realisation of that point")
    parser.add_argument("--noise-free", action="store_true", help="analyse the noise-free model of the point")
    parser.add_argument("--observables", choices=("both", "nc", "gt"), default="both", help="data used in the likelihood")
    parser.add_argument("--cov", choices=("fiducial", "truth", "varying"), default="fiducial",
                        help="covariance at the fiducial parameters, at the true parameters, or at each step")
    parser.add_argument("--fix", default="", help="parameters or groups (cosmo, hod, misc, bsys) fixed to the true value")
    parser.add_argument("--calibration", default=gcm.DEFAULT_CALIBRATION, help="calibration file")
    parser.add_argument("--nlive", type=int, default=500, help="dynesty live points")
    parser.add_argument("--dlogz", type=float, default=0.1, help="dynesty stopping criterion")
    parser.add_argument("--sample", default="rwalk", help="dynesty sampling method")
    parser.add_argument("--maxiter", type=int, default=None, help="maximum number of iterations (tests)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the sampler")
    parser.add_argument("--ncpu", type=int, default=1, help="number of processes")
    parser.add_argument("--output", required=True, help="output file (.npz); checkpoint <output>.save")
    args = parser.parse_args()

    # --- data and model settings from the mock file
    gcm.load_calibration(args.calibration)
    names = list(gcm.PARAMETERS)
    mock = np.load(args.data)
    if list(mock["param_names"]) != names:
        sys.exit(f"{args.data}: its {len(mock['param_names'])} parameters differ from the {len(names)} of the model "
                 f"({', '.join(names)}): made with another version of generate_cluster_mocks.py?")
    cal = gcm.Calibration(args.calibration)
    for key, edges in (("z_obs_edges", cal.z_obs_edges), ("lambda_obs_edges", cal.lambda_obs_edges),
                       ("r_edges_comoving", cal.r_edges)):
        if not np.allclose(mock[key], edges):
            sys.exit(f"{args.data}: {key} differ from the calibration file")
    row = int(np.nonzero(mock["index"] == args.index)[0][0]) if "index" in mock.files else args.index
    theta_true = mock["theta"][row]
    if "ok" in mock.files and not mock["ok"][row]:
        sys.exit(f"{args.data}: the model failed at point {args.index}")
    want_nc, want_gt = args.observables in ("both", "nc"), args.observables in ("both", "gt")
    key = (lambda k: f"{k}_mean") if args.noise_free else (lambda k: k)
    data = {}
    for obs, want in (("nc", want_nc), ("gt", want_gt)):
        if not want:
            data[obs] = None
            continue
        if key(obs) not in mock.files:
            made = str(mock["observables"]) if "observables" in mock.files else "both"
            sys.exit(f"{args.data} has no {obs} data (made with --observables {made})")
        data[obs] = mock[key(obs)][row] if args.noise_free else mock[obs][row, args.real]
    miscentring = bool(mock["miscentring"]) if "miscentring" in mock.files else True
    settings = dict(area=float(mock["area"]), n_eff=float(mock["n_eff"]), sigma_e=float(mock["sigma_e"]),
                    misc_profile=str(mock["misc_profile"]), miscentring=miscentring)

    # --- fixed parameters
    fixed = gcm.expand_fixed(args.fix)
    if args.observables == "nc":  # miscentring and B(R) only enter the lensing
        fixed |= set(gcm.PARAMETER_GROUPS["misc"]) | set(gcm.PARAMETER_GROUPS["bsys"])
    if not miscentring:
        fixed |= set(gcm.PARAMETER_GROUPS["misc"])
    free = [p for p in names if p not in fixed]

    # --- covariances (fixed ones computed here, once)
    t0 = time.time()
    model = gcm.ClusterModel(cal, observables=args.observables, **settings)
    cov = None
    if args.cov != "varying":
        theta_cov = np.array([gcm.PARAMETERS[p][0] for p in names]) if args.cov == "fiducial" else theta_true
        warnings.filterwarnings("ignore")
        _, _, cov_nc, cov_gt = model(theta_cov, with_covariance=True)
        cov = (cov_nc, cov_gt)
        print(f"Covariances at the {args.cov} parameters: {time.time() - t0:.1f} s", flush=True)

    cfg = dict(calibration=args.calibration, observables=args.observables, fixed=sorted(fixed),
               theta_true=theta_true, nc_data=data["nc"], gt_data=data["gt"], cov=cov, **settings)
    setup(cfg)
    truth = theta_true[S["i_free"]]
    t1 = time.time()
    logl_true = loglike(truth)
    print(f"Data: {args.data}, point {args.index}, " + ("noise-free" if args.noise_free else f"realisation {args.real}")
          + f"; observables {args.observables}; covariance {args.cov}")
    print(f"Free parameters ({len(free)}): {', '.join(free)}")
    print(f"ln L at the truth = {logl_true:.3f}  ({time.time() - t1:.2f} s per call)", flush=True)

    # --- nested sampling
    checkpoint = args.output + ".save"
    run_kwargs = dict(dlogz=args.dlogz, checkpoint_file=checkpoint, maxiter=args.maxiter)
    # "spawn": JAX does not work in forked processes; each worker builds its own model (setup)
    pool = mp.get_context("spawn").Pool(args.ncpu, initializer=setup, initargs=(cfg,)) if args.ncpu > 1 else None
    pool_kwargs = dict(pool=pool, queue_size=args.ncpu) if pool is not None else {}
    t_start = time.time()
    try:
        if os.path.exists(checkpoint):
            print(f"Resuming from {checkpoint}")
            sampler = dynesty.NestedSampler.restore(checkpoint, pool=pool)
            if sampler.added_live:  # finished run: remove the final live points to continue
                sampler._remove_live_points()
            sampler.run_nested(resume=True, **run_kwargs)
        else:
            sampler = dynesty.NestedSampler(
                loglike, prior_transform, len(free), nlive=args.nlive, bound="multi", sample=args.sample,
                rstate=np.random.default_rng(args.seed), **pool_kwargs,
            )
            sampler.run_nested(**run_kwargs)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"\nCompleted in {(time.time() - t_start) / 60:.1f} min")

    # --- output
    res = sampler.results
    weights = np.exp(res.logwt - res.logz[-1])
    weights /= weights.sum()
    fiducial = np.array([gcm.PARAMETERS[p][0] for p in free])
    out = dict(
        samples=res.samples, weights=weights, logl=res.logl, logz=res.logz[-1], logzerr=res.logzerr[-1],
        param_names=np.array(free), truth=truth, fiducial=fiducial, all_param_names=np.array(names),
        theta_true=theta_true, fixed=np.array(sorted(fixed)), data_file=os.path.abspath(args.data),
        index=args.index, real=args.real, noise_free=args.noise_free, observables=args.observables,
        cov_mode=args.cov, nlive=args.nlive, dlogz=args.dlogz, command=" ".join(sys.argv), **settings,
    )
    for k, v in dict(nc_data=data["nc"], gt_data=data["gt"]).items():
        if v is not None:
            out[k] = v
    if cov is not None:
        for k, v in zip(("cov_nc", "cov_gt"), cov):
            if v is not None:
                out[k] = v
    np.savez(args.output, **out)

    mean, c = dyfunc.mean_and_cov(res.samples, weights)
    std = np.sqrt(np.diag(c))
    print(f"N_eff = {1.0 / np.sum(weights**2):.0f}, ln Z = {res.logz[-1]:.2f} +/- {res.logzerr[-1]:.2f}, "
          f"best ln L = {res.logl.max():.3f}")
    print(f"  {'param':12s} {'truth':>10s} {'mean':>10s} {'std':>10s} {'(mean-truth)/std':>17s}")
    for i, p in enumerate(free):
        print(f"  {p:12s} {truth[i]:10.4f} {mean[i]:10.4f} {std[i]:10.4f} {(mean[i] - truth[i]) / std[i]:17.2f}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
