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
git clone git@github.com:m-aguena/cloelib.git
cd cloelib
git checkout sbi-mocks-v1
pip install -e .
pip install cosmopower-jax jax camb  # if not already installed
```

The power-spectrum emulators (~10 MB) are downloaded from Zenodo the first time the script runs. They
are saved in `cloelib/cosmology/emulator-data-jax/`.

## Usage

```bash
cd scripts/cluster_sbi_mocks
python generate_cluster_mocks.py --list-params           # parameters, fiducial values, priors
python generate_cluster_mocks.py --n-points 1000 --ncpu 8 --output mocks_prior.npz
python generate_cluster_mocks.py --mode fiducial --n-real 200 --output mocks_fiducial.npz
python generate_cluster_mocks.py --n-points 1000 --fix misc,pca --output mocks_fixed_nuisances.npz
python generate_cluster_mocks.py --help                   # all the options
```

There are 41 free parameters:

| Group | Parameters | Prior |
|---|---|---|
| Cosmology | Ω_m, σ_8 | uniform |
| Richness–mass relation | log10 M_1, α, ε, σ_lnλ | uniform |
| Miscentring | f, s_1, s_2 | Gaussian |
| PCA amplitudes | a_1, a_2 per bin (32 in all) | Gaussian, from the PICCOLO calibration |

The fiducial values are the PICCOLO truth and calibrations. `--fix` takes parameter names or groups
(`cosmo`, `hod`, `misc`, `pca`) and keeps them at their fiducial values. When the PCA amplitudes are
fixed, their calibration uncertainty is added to the g_t noise.

Cost: about 5–30 s per point and process, depending on machine load. Most of it is the lensing
model. The output is written every `--checkpoint` points. Use `--resume` to continue an interrupted
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

## Files

- `generate_cluster_mocks.py`: the script.
- `calibration_A1900_apLtr_perc.npz`: PICCOLO calibrations of the A1900 setup (from the even patches
  of the percolated catalogues), containing:
  - the selection-function best fit;
  - the comoving 2-Rayleigh miscentring;
  - the PCA template B̄(R), the modes e_k(R), and the amplitudes per bin with their covariance.
