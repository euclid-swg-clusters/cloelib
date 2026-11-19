# Mock cluster observables for SBI tests

`generate_cluster_mocks.py` produces mock Euclid-like cluster observables with the cloelib cluster
model, for testing simulation-based inference (SBI) pipelines:

- **NC**: number counts in 4 observed-redshift bins (0.2–1.0) × 4 observed-richness bins (40–250).
- **g_t**: stacked reduced tangential shear profiles in the same bins, with 9 comoving radial bins
  (0.25–9 Mpc/h).

Noise is drawn from the cloelib covariances, evaluated at each parameter point:

- NC: Poisson + sample variance.
- g_t: shape noise + LSS + intrinsic scatter.

The model is the one of the Euclid DR1 cluster "A1900" tests (PICCOLO mock, 1900 deg²). It uses:

- shifted-Poisson richness–mass relation;
- analytic (Costanzi+19) selection function with Gaussian photo-z;
- BMO profiles with a 2-halo term with halo exclusion;
- 2-Rayleigh miscentring in comoving units;
- PCA correction of the optical selection bias.

The script's docstring and comments describe the model in detail.

## Installation

```bash
git clone --branch sbi-mocks-v1 https://github.com/euclid-swg-clusters/cloelib.git
cd cloelib
python -m venv ../venv-cloelib && source ../venv-cloelib/bin/activate   # optional, recommended
pip install -e .
pip install cosmopower-jax camb astropy   # used by the cluster modules, not installed by cloelib
```

Python >= 3.10 is required.

The power-spectrum emulators (~10 MB) are downloaded from Zenodo the first time the script runs. They
are saved in `cloelib/cosmology/emulator-data-jax/`.

## Usage

```bash
cd scripts/cluster_sbi_mocks
python generate_cluster_mocks.py --list-params           # parameters, fiducial values, priors
python generate_cluster_mocks.py --n-points 1000 --ncpu 8 --output mocks_prior.npz
python generate_cluster_mocks.py --mode fiducial --n-real 200 --output mocks_fiducial.npz
python generate_cluster_mocks.py --n-points 1000 --fix misc,bsys --output mocks_fixed_nuisances.npz
python generate_cluster_mocks.py --n-points 10000 --observables nc --output mocks_nc.npz   # NC only
python generate_cluster_mocks.py --n-points 10000 --observables gt --output mocks_gt.npz   # g_t only
python generate_cluster_mocks.py --help                   # all the options
```

`--observables` selects what is computed: `both` (default), `nc` (number counts only) or `gt` (reduced
shear only). The observable that is not requested, and its covariance, are not computed or saved.
With `nc` the miscentring and B-systematics parameters are fixed, since they only enter the lensing.
For the same `--seed`, the cosmology and richness–mass parameters of each point are the same whatever
the choice.

There are 12 free parameters:

| Group | Parameters | Prior |
|---|---|---|
| Cosmology | Ω_m, σ_8 | uniform |
| Richness–mass relation | log10 M_1, α, ε, σ_lnλ | uniform |
| Miscentring | f, s_1, s_2 | Gaussian, truncated |
| Optical selection bias systematics | B_s, B_da1, B_da2 | Gaussian, mean 0 |

The fiducial values are the PICCOLO truth and calibrations (0 for the B systematics). `--fix` takes
parameter names or groups (`cosmo`, `hod`, `misc`, `bsys`) and keeps them at their fiducial values.

The optical selection bias correction is B(R) = B̄(R) + Σ_k a_k e_k(R), with two PCA amplitudes a_k
per bin. The amplitudes are fixed to their PICCOLO calibration, and their statistical uncertainty is
added to the g_t noise. The systematic uncertainty of the calibration is described by 3 parameters
shared by all the bins:

B(R) − 1 = (1 + B_s) [B̄(R) − 1 + Σ_k (a_k + B_da_k) e_k(R)]

Cost: about 4 s per point and process on an idle CPU core, plus ~10 s of start-up per process. The
model takes ~2 s (half of it the miscentred profiles) and the covariances ~2 s. Two options, both off
by default, make it faster:

| Options | Time per point |
|---|---|
| default | ~4 s |
| `--cov-fiducial` | ~2 s |
| `--no-miscentring` | ~3 s |
| `--cov-fiducial --no-miscentring` | ~1 s |
| `--observables nc` | ~0.3 s |
| `--observables gt` | ~3.6 s (~1.8 s with `--cov-fiducial`) |

- `--cov-fiducial` computes the noise covariances once, at the fiducial point, and uses them for all
  the points. They are saved in the output as `cov_nc_fid` and `cov_gt_fid`.
- `--no-miscentring` centres all the clusters and fixes the miscentring parameters. g_t then changes
  by up to ~80% in some bins, and the B(R) calibration assumed miscentring.

The output is written every `--checkpoint` points. Use `--resume` to continue an interrupted
run, or to extend it with a larger `--n-points`: the parameter points and the noise are reproducible
from `--seed`.

## Output

The output is an `.npz` file. Its main arrays:

- `theta`, `param_names`: the parameter points.
- `nc_mean`, `gt_mean`: noise-free observables.
- `nc`, `gt`: noisy realisations, with shape (n_points, n_real, ...).
- `seed`: the noise seed of each point.
- `ok`: False where the model failed.
- `cov_nc`, `cov_gt`: the covariances, with `--save-cov`.

The file also stores the bin edges and the run settings. The module docstring describes every array.

## Likelihood inference (reference posteriors)

`run_chain.py` runs nested sampling (dynesty) on one data vector of a mock file. It uses the same
model, parameters and priors as the generator, so its posterior is the reference to compare SBI with.
dynesty must be installed (`pip install dynesty`).

```bash
python generate_cluster_mocks.py --mode fiducial --n-real 1 --output mock_fid.npz
python run_chain.py --data mock_fid.npz --ncpu 16 --output chain_fid.npz
python run_chain.py --data mock_fid.npz --observables nc --ncpu 16 --output chain_fid_nc.npz
python run_chain.py --data mocks_prior.npz --index 12 --real 3 --cov truth --output chain_12.npz
```

- `--index`, `--real`: the point and the noise realisation of the mock file to analyse.
  `--noise-free` analyses the noise-free model of that point instead.
- `--observables`: `both`, `nc` or `gt`.
- `--cov`: the Gaussian likelihood uses the cloelib covariances, computed at the fiducial parameters
  (`fiducial`, the default, as in the DR1 chains), at the true parameters of the mock point (`truth`),
  or at each likelihood call (`varying`, with the ln det term, about twice slower).
  - Mocks made with `--cov-fiducial` have noise drawn with the fiducial covariance: use `--cov fiducial`.
  - The default mocks have noise drawn with the covariance of each point: use `--cov truth` or
    `--cov varying`.
- `--fix`: parameters or groups fixed to their true values.
- The survey area, source density, shape noise and miscentring settings are read from the mock file.
- An interrupted chain resumes from its checkpoint (`<output>.save`) when the same command is rerun.

A chain needs about 1e5 likelihood calls (12 parameters, 500 live points). A call takes ~2 s for
NC + g_t and ~0.25 s for NC only, so a chain takes a few hours (NC + g_t) or about 30 min (NC only) on
16 cores.

The output `.npz` has the weighted samples (`samples`, `weights`), `logl`, `logz`, the sampled
`param_names` and their `truth`, the data and the covariances. The script prints the posterior means
and their offsets from the truth.

## Files

- `generate_cluster_mocks.py`: the mock generator.
- `run_chain.py`: likelihood inference with dynesty on a mock.
- `calibration_A1900_apLtr_perc.npz`: PICCOLO calibrations of the A1900 setup (from the even patches
  of the percolated catalogues), containing:
  - the selection-function best fit;
  - the comoving 2-Rayleigh miscentring;
  - the PCA template B̄(R), the modes e_k(R), and the amplitudes per bin with their covariance.
