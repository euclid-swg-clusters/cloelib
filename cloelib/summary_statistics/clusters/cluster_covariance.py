"""Cluster covariance calculations."""

import numpy as np
from scipy.integrate import simpson
from scipy.special import eval_legendre, j0, j1, sici, spherical_jn

from cloelib.auxiliary import units
from cloelib.auxiliary.halo_helpers import photoz_rsd_monopole_correction
from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations
from .counts import ClusterCounts
from .clustering import ClusterClustering
from .lensing import ClusterWeakLensing


class ClusterCovariance:
    """
    Compute the count or clustering-monopole covariance.

    The summary statistic type determines the calculation used by `compute`.
    Predictions and intermediate products must come from the same model and bins.
    """

    def __init__(self, summary_statistic, z_tab_integ=31, ell_max=20):
        """
        Configure covariance calculations for a summary statistic instance.

        Parameters
        ----------
        summary_statistic : ClusterCounts or ClusterClustering
            Supplies the shared integration model, cosmology, k grid and survey
            area. Construct this object, then call `compute` with the
            prediction's intermediate products.
        z_tab_integ : int, optional
            Radial integration points for the count survey window. Default: 31.
        ell_max : int, optional
            Maximum angular multipole of the count survey window. Default: 20.
            This is not the correlation-function multipole.

        See Also
        --------
        compute : Select and evaluate the appropriate covariance.
        """
        if isinstance(summary_statistic, ClusterCounts):
            self._cov_type = "counts"
        elif isinstance(summary_statistic, ClusterClustering):
            self._cov_type = "clustering"
        elif isinstance(summary_statistic, ClusterWeakLensing):
            raise NotImplementedError(
                "Use ClusterWeakLensingCovariance for the weak-lensing covariance."
            )
        else:
            raise TypeError("Expected ClusterCounts or ClusterClustering.")
        if not isinstance(z_tab_integ, (int, np.integer)) or z_tab_integ < 2:
            raise ValueError("z_tab_integ must be an integer >= 2.")
        if not isinstance(ell_max, (int, np.integer)) or ell_max < 0:
            raise ValueError("ell_max must be a nonnegative integer.")

        self.cluster_statitstics_modeling = (
            summary_statistic.cluster_statitstics_modeling
        )
        self.ell_max = ell_max
        self.z_tab_integ = z_tab_integ
        self.rint = None

    def compute(
        self,
        *,
        intermediates,
        prediction=None,
        z_obs_edges=None,
        lambda_obs_edges=None,
    ):
        """
        Select the covariance from the supplied summary statistic type.

        Parameters
        ----------
        intermediates : dict
            Products returned by get_NC or get_xi0 from the same model and bins.
            Counts require window_z_lambda_obs. Clustering
            requires pk0_mean_values, radial_shell_window, radial_shell_volume,
            window_z_obs and cluster_counts. Higher multipoles are not supported.
        prediction : numpy.ndarray, optional
            Count prediction of shape (z_obs_bins, richness_bins), required for
            ClusterCounts. Clustering uses intermediates['cluster_counts'].
        z_obs_edges : numpy.ndarray, optional
            Observed-redshift bin edges from get_NC, required for ClusterCounts.
        lambda_obs_edges : numpy.ndarray, optional
            Observed-richness bin edges from get_NC, required for ClusterCounts.

        Returns
        -------
        numpy.ndarray
            Counts: (z_obs, z_obs, richness, richness).
            Clustering: (z_obs, z_obs, richness_pair, richness_pair, radius, radius).
            These are covariance tensors, not flattened two-dimensional matrices.

        Examples
        --------
        Given a configured counts model and its bin edges:

        >>> prediction, products = counts.get_NC(z_edges, richness_edges)
        >>> covariance = ClusterCovariance(counts)
        >>> matrix = covariance.compute(
        ...     prediction=prediction,
        ...     z_obs_edges=z_edges,
        ...     lambda_obs_edges=richness_edges,
        ...     intermediates=products,
        ... )

        For a configured clustering model:

        >>> xi, products = clustering.get_xi0(z_edges, richness_edges, radius_edges)
        >>> matrix = ClusterCovariance(clustering).compute(intermediates=products)
        """
        if self._cov_type == "counts":
            if prediction is None or z_obs_edges is None or lambda_obs_edges is None:
                raise ValueError(
                    "Counts require prediction, z_obs_edges and lambda_obs_edges."
                )
            z_obs_edges = np.asarray(z_obs_edges)
            lambda_obs_edges = np.asarray(lambda_obs_edges)
            prediction = np.asarray(prediction)
            if (
                z_obs_edges.ndim != 1
                or z_obs_edges.size < 2
                or not np.all(np.isfinite(z_obs_edges))
                or np.any(np.diff(z_obs_edges) <= 0)
            ):
                raise ValueError(
                    "Redshift edges must be finite and strictly increasing."
                )
            if prediction.ndim != 2 or prediction.shape[0] != z_obs_edges.size - 1:
                raise ValueError("Count shape does not match the redshift bins.")
            if (
                lambda_obs_edges.ndim != 1
                or prediction.shape[1] != lambda_obs_edges.size - 1
            ):
                raise ValueError("Count shape does not match the richness bins.")
            # The spatial-window loop fills this workspace in increasing bin order.
            self.rint = np.zeros(
                (
                    z_obs_edges.size - 1,
                    len(self.cluster_statitstics_modeling.tabulated_integrands["k"]),
                    self.ell_max + 1,
                )
            )
            return self._get_NC_covariance(
                z_obs_edges,
                lambda_obs_edges,
                prediction,
                intermediates["window_z_lambda_obs"],
            )
        if "pk0_mean_values" not in intermediates:
            raise NotImplementedError(
                "Pass get_xi0 products; higher-multipole covariance is not implemented."
            )
        return self._get_xi_covariance(
            intermediates["pk0_mean_values"],
            intermediates["radial_shell_window"],
            intermediates["radial_shell_volume"],
            intermediates["window_z_obs"],
            intermediates["cluster_counts"],
        )

    def _Kl_coeff(self):
        """
        Coefficients of the spherical harmonics expansion of the angular part of the window function

        Parameters
        ----------
        L: int
           Maximum number at which to evaluate the coefficients

        Returns
        -------
        KL: numpy.ndarray
            Coefficients up to L multipole
        """

        ell = np.linspace(0, self.ell_max, self.ell_max + 1, dtype=int)

        area = self.cluster_statitstics_modeling.area
        theta = np.arccos(1 - (area * (np.pi / 180.0) ** 2.0) / (2 * np.pi))

        KL = (
            np.sqrt(np.pi / (2.0 * ell + 1.0))
            * (
                eval_legendre(ell - 1, np.cos(theta))
                - eval_legendre(ell + 1, np.cos(theta))
            )
            / (2.0 * np.pi * (1 - np.cos(theta)))
        )

        KL[0] = 1 / (2.0 * np.sqrt(np.pi))

        return KL

    def _cov_window(self, iz, z_window_edges, KL):
        """
        Computes the window function between redshifts bins

        Parameters
        ----------
        iz: int
            Index of the redshift bins at which to evaluate the window function
        z_window_edges: tuple
             Redshift boundaries (min, max values) of the window.
        KL: numpy.ndarray
            Spherical harmonic expansion coefficients

        Returns
        -------
        cluster count covariance window:   numpy.ndarray
                W[i,j,k] where i and j are two redshift bin and k are the wavenumbers
        """
        zarr_iz = np.linspace(z_window_edges[0], z_window_edges[1], self.z_tab_integ)

        background = self.cluster_statitstics_modeling.halo_model_properties.background

        rvec = background.comoving_distance(zarr_iz) * (
            background.H0 / 100.0
        )  # Mpc h^{-1}

        Vz = (rvec[-1] ** 3 - rvec[0] ** 3) / 3  # Mpc^3 h^{-3}

        kr = (
            self.cluster_statitstics_modeling.tabulated_integrands["k"][:, np.newaxis]
            * rvec
        )

        self.rint[iz] = (
            1
            / Vz
            * simpson(
                rvec**2.0
                * np.array(
                    [
                        spherical_jn(_l, kr, derivative=False)
                        for _l in range(self.ell_max + 1)
                    ]
                ),
                x=rvec,
                axis=-1,
            ).T
        )
        return (4 * np.pi) * np.sum(
            self.rint[iz, :, :] * self.rint[: (iz + 1), :, :] * KL[:] ** 2, axis=-1
        )

    def _compute_spatial_cov(self, z_obs_edges, lambda_obs_mean):
        """Computes only spatial part of the covariance.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_mean : numpy.ndarray
            Mean observed richness of the clusters in each redshift bin, at which the
            scatter of the observed redshift is evaluated. Dimensions: (z_obs).

        Returns
        -------
        spatial_cov : numpy.ndarray
            Spatial part of the covariance. Dimensions: (z_obs, z_obs).
        """

        z_obs_edges_size = len(z_obs_edges) - 1
        z_mid = 0.5 * (z_obs_edges[1:] + z_obs_edges[:-1])

        # power spectrum at the center of observed redshift bins (z_obs, k)
        pk = self.cluster_statitstics_modeling.halo_model_properties.matter_power_spectrum_cb(
            z_mid, self.cluster_statitstics_modeling.tabulated_integrands["k"]
        )

        # corrected halo Pk (only 0-th order correction is enough for number counts covariance)
        # neglecting the richness dependence of the scatter within each redshift bin
        pk *= photoz_rsd_monopole_correction(
            self.cluster_statitstics_modeling.halo_model_properties.background,
            z_mid,
            self.cluster_statitstics_modeling.tabulated_integrands["k"],
            self.cluster_statitstics_modeling.selection_function.scatter_z_obs(
                lambda_obs_mean, z_mid
            ),
        )[0]

        # spherical harmonic expansion coefficients (covariance)
        KL = self._Kl_coeff()

        # compute spatial covariance (z_obs, z_obs)
        spatial_cov = np.zeros((z_obs_edges_size, z_obs_edges_size))
        for ind_z in range(z_obs_edges_size):
            spatial_cov[ind_z, : (ind_z + 1)] = (
                self.cluster_statitstics_modeling.integrate_probe_function_in_dk(
                    np.sqrt(pk[ind_z] * pk[: (ind_z + 1)])
                    * self._cov_window(
                        ind_z,
                        (
                            z_obs_edges[ind_z],
                            z_obs_edges[ind_z + 1],
                        ),
                        KL,
                    ),
                )
            )
            # fill 2nd half of symmetrical matrix
            spatial_cov[: (ind_z + 1), ind_z] = spatial_cov[ind_z, : (ind_z + 1)]
        return spatial_cov

    def _get_NC_covariance(
        self, z_obs_edges, lambda_obs_edges, cluster_counts, window_z_lambda_obs
    ):
        """Computes theoretical covariance for cluster counts, including shot noise and sample covariance

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        cluster_counts : numpy.ndarray
            Number counts in redshift and richness bins
        window_z_lambda_obs : numpy.ndarray
            P(lamda_obs_bin,z_obs_bin|M, ztrue).
            Dimensions: (z_obs, lambda_obs, ztrue, M) with (ztrue, M) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_NC.

        Returns
        -------
        cov_cluster_counts : numpy.ndarray
            Covariance number counts in redshift and richness bins
        """

        ############################################
        # Get cluster statistics modeling quantities
        ############################################

        # integral of P(lambda_obs|M, z)*dn/dM*bias on lambda_obs bins and mass : (lambda_obs, ztrue)
        # cluster integrated bias : (z_obs, lambda_obs)
        halo_bias_mean_values = self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
            # integral of P(lambda_obs_bin,z_obs_bin|M, z)*dn/dM on lambda_obs bins and mass : (lambda_obs, ztrue)
            self.cluster_statitstics_modeling.integrate_probe_function_in_mass(
                self.cluster_statitstics_modeling.tabulated_integrands["bias(ztrue,M)"],
                window_z_lambda_obs,
            ),
        )

        ####################
        # Compute covraiance
        ####################

        # mean richness of each redshift bin, from the centers of the richness bins
        # weighted by the counts (simple average if there are no clusters) : (z_obs)
        lambda_mid = 0.5 * (lambda_obs_edges[1:] + lambda_obs_edges[:-1])
        total_counts = cluster_counts.sum(axis=1)
        lambda_obs_mean = np.divide(
            np.sum(cluster_counts * lambda_mid, axis=1),
            total_counts,
            out=np.full(total_counts.shape, lambda_mid.mean()),
            where=total_counts > 0.0,
        )

        # spatial component of covariance (z_obs, z_obs)
        spatial_cov = self._compute_spatial_cov(z_obs_edges, lambda_obs_mean)

        # shot noise (z_obs, z_obs, lambda_obs, lambda_obs)
        _shot_noise = (
            np.diag(cluster_counts.flatten())
            .reshape(*cluster_counts.shape, *cluster_counts.shape)
            .transpose(0, 2, 1, 3)
        )

        # total covariance = shot-noise + sample covariance (z_obs, z_obs, lambda_obs, lambda_obs)
        cov_cluster_counts = _shot_noise + (
            halo_bias_mean_values[np.newaxis, :, np.newaxis, :]
            * halo_bias_mean_values[:, np.newaxis, :, np.newaxis]
            * spatial_cov[:, :, np.newaxis, np.newaxis]
        )

        return cov_cluster_counts

    def _get_xi_covariance(
        self,
        pk_mean_values,
        radial_shell_window,
        radial_shell_volume,
        window_z_obs,
        cluster_counts,
    ):
        """Computes clustering covariance.

        Parameters
        ----------
        pk_mean_values : numpy.ndarray
            Power spectrum averaged on redshift and richnesses bins (with IR-resummation).
            Is in the intermediate_integration_products output of get_xi0.
        radial_shell_window : numpy.ndarray
            Cluster count covariance window (z_obs, lambda_obs, k),
            with (k) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_xi0.
        radial_shell_volume : numpy.ndarray
            Spherical shell volume (z_obs, radius).
            Is in the intermediate_integration_products output of get_xi0.
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, ztrue) in z_obs bins.
            Dimensions: (z_obs, lambda_obs, ztrue) with (ztrue) in cluster_statitstics_modeling.tabulated_integrands.
            Is in the intermediate_integration_products output of get_xi0.
        cluster_counts : numpy.ndarray
            Number counts in redshift and richness bins

        Returns
        -------
        cov_cluster_clustering : numpy.ndarray
            Covariance of the two point correlation function in richness, redshift and radial bins
        """
        z_obs_edges_size, lambda_obs_edges_size = cluster_counts.shape
        _, radius_edges_size = radial_shell_volume.shape

        ########################################
        # Cluster statistics modeling quantities
        ########################################

        # Compute observed volume in each redshift bin : (z_obs, lambda_obs)
        volume_mean_values = (
            self.cluster_statitstics_modeling.integrate_probe_function_in_redshift(
                window_z_obs
            )
        )
        # Compute output shot-noise terms : (z_obs, lambda_obs, lambda_obs)
        vol_over_cluster_counts = (
            volume_mean_values[:, :, np.newaxis]
            / cluster_counts[:, :, np.newaxis]
            * np.identity(lambda_obs_edges_size)[np.newaxis, :, :]
        )

        #############################
        # Compute nuisance parameters
        #############################

        #    alpha(ztrue,lambda_obs), beta(ztrue,lambda_obs),
        #    gamma(ztrue,lambda_obs) are nuisance parameters to be fitted on
        #    (few, ~100) simulations to correct for bias model inaccuracy,
        #    non-poissonian shot-noise and high-order terms ref values are
        #    alpha=0, beta=1, gamma=0 (see Euclid Collaboration : Fumagalli et
        #    al. 2022)
        alpha = np.zeros((z_obs_edges_size, lambda_obs_edges_size))
        beta = np.ones((z_obs_edges_size, lambda_obs_edges_size))
        gamma = np.zeros((z_obs_edges_size, lambda_obs_edges_size))

        # Combine alpha, beta with pk, vol and reshape to be used
        # in cov_g, cov_ng integral
        # shape (z_obs_edges, lambda_obs, lambda_obs, k)
        beta_pk_mean_values = (
            beta[:, :, np.newaxis, np.newaxis]
            * beta[:, np.newaxis, :, np.newaxis]
            * pk_mean_values
        )
        # shape (z_obs_edges, lambda_obs, lambda_obs, k)
        avol_bpk_mean_values = (
            # reshape alpha to be (z_obs, lambda_obs, lambda_obs)
            (1 + alpha)[:, :, np.newaxis, np.newaxis]
            * (1 + alpha)[:, np.newaxis, :, np.newaxis]
            * vol_over_cluster_counts[:, :, :, np.newaxis]
            + beta_pk_mean_values
        )

        ####################
        # Compute covariance
        ####################

        # define cluster clustering bin numbers for loops
        lambda_bin_loop = range(lambda_obs_edges_size)
        rad_bin_loop = range(radius_edges_size)

        # cov_g, cov_ng are TWO TERMS OF EQ. 73
        _cov_gaussian = np.zeros(
            (
                z_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                radius_edges_size,
                radius_edges_size,
            )
        )
        _cov_nongaussian = np.zeros(
            (
                z_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                lambda_obs_edges_size,
                radius_edges_size,
                radius_edges_size,
            )
        )

        # note : this could be reduced to compute only half of the matrix
        for ind_lambda_i in lambda_bin_loop:
            for ind_lambda_j in lambda_bin_loop:
                for ind_radius in rad_bin_loop:
                    _cov_nongaussian[
                        :,
                        ind_lambda_i,
                        ind_lambda_j,
                        ind_lambda_i,
                        ind_lambda_j,
                        ind_radius,
                        ind_radius,
                    ] = (
                        self.cluster_statitstics_modeling.integrate_probe_function_in_dk(
                            radial_shell_window[:, ind_radius, :]
                            * beta_pk_mean_values[:, ind_lambda_i, ind_lambda_j, :],
                        )
                        * (1 + gamma[:, ind_lambda_i])
                        * vol_over_cluster_counts[:, ind_lambda_i, ind_lambda_i]
                        * (1 + gamma[:, ind_lambda_j])
                        * vol_over_cluster_counts[:, ind_lambda_j, ind_lambda_j]
                        / radial_shell_volume[:, ind_radius]
                    )

                for ind_lambda_k in lambda_bin_loop:
                    for ind_lambda_h in lambda_bin_loop:
                        # somehow using integrate_probe_function_in_k is much faster then
                        # integrate_probe_function_in_dk here, to be investigated

                        # gaussian term
                        _cov_gaussian[
                            :,
                            ind_lambda_i,
                            ind_lambda_j,
                            ind_lambda_k,
                            ind_lambda_h,
                            :,
                            :,
                        ] = self.cluster_statitstics_modeling.integrate_probe_function_in_k(
                            radial_shell_window[:, np.newaxis, :, :]
                            * radial_shell_window[:, :, np.newaxis, :]
                            * avol_bpk_mean_values[
                                :, ind_lambda_i, ind_lambda_k, np.newaxis, np.newaxis, :
                            ]
                            * avol_bpk_mean_values[
                                :, ind_lambda_j, ind_lambda_h, np.newaxis, np.newaxis, :
                            ]
                            * self.cluster_statitstics_modeling.tabulated_integrands[
                                "dk"
                            ],
                        )

        # Compute the covariance : (z_obs, lambda_obs,  lambda_obs, lambda_obs, lambda_obs, radius, radius)
        # for tranposing lambda_obs_clustering bins
        _invert_index = (0, 1, 2, 4, 3, 5, 6)
        _cov_clustering_4_lambda_obs_edges = (
            (_cov_gaussian + _cov_nongaussian)
            + (_cov_gaussian + _cov_nongaussian).transpose(_invert_index)
        ) / volume_mean_values[
            :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]

        # cov_xi(lambda_obs_i, lambda_obs_j, lambda_obs_k, lambda_obs_l) =
        # cov_xi(lambda_obs_j, lambda_obs_i, lambda_obs_l, lambda_obs_k)
        # so reshape and keep only two of them
        triangle_indexes = np.triu_indices(lambda_obs_edges_size)
        # simplify first pair
        _cov_clustering_3_lambda_obs_edges = _cov_clustering_4_lambda_obs_edges[
            :, triangle_indexes[0], triangle_indexes[1], :, :, :, :
        ]
        # simplify second pair
        _cov_clustering_2_lambda_obs_edges = _cov_clustering_3_lambda_obs_edges[
            :, :, triangle_indexes[0], triangle_indexes[1], :, :
        ]

        ### EQ. 89 + RESHAPE according to 2ptCF ###
        # Current covariance is shape (z_obs, lambda_obs, lambda_obs, radius, radius),
        # make it (z_obs, z_obs, lambda_obs, lambda_obs, radius, radius),
        # being diagonal in (z_obs, z_obs)
        cov_cluster_clustering = (
            np.identity(z_obs_edges_size)[
                :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
            ]
            * _cov_clustering_2_lambda_obs_edges
        )
        return cov_cluster_clustering


class ClusterWeakLensingCovariance:
    def __init__(
        self,
        cluster_weak_lensing: ClusterWeakLensing,
        n_eff=15.0,
        sigma_e: float = 0.3,
        nonlinear_perturbations: Perturbations = None,
        source_delta_z: float = 0.05,
        source_selection: str = "z_true",
        ell: np.ndarray = np.geomspace(1.0, 2.0e5, 1500),
        z_los: np.ndarray = None,
        z_sources_size: int = 200,
        k_nonlinear: np.ndarray = np.geomspace(1.0e-4, 1.0e2, 400),
    ):
        r"""
        Initializes the Gaussian covariance of the cluster tangential shear profile,
        following Wu et al. 2019 (https://arxiv.org/abs/1907.06611), Eq. 10:

        ..math:
            {\rm Cov}[\gamma_t^i(\theta_a), \gamma_t^j(\theta_b)] =
            \frac{1}{4\pi f_{\rm sky}}\int\frac{\ell d\ell}{2\pi}
            \hat J_2(\ell\theta_a)\hat J_2(\ell\theta_b)
            \left[\left(C_\ell^{h_ih_j}+\frac{\delta_{ij}}{n_h^i}\right)
            \left(C_\ell^{\kappa\kappa}+\frac{\sigma_e^2}{n_s}\right)
            + C_\ell^{h_i\kappa}C_\ell^{h_j\kappa}\right]

        where i, j are observed richness bins in the same observed redshift bin.
        The lens sample (redshift distribution, surface density, bias and 1-halo
        term) is predicted from the halo mass function and the selection function
        of the cluster statistics modeling, and the source sample from the
        tomographic source redshift distribution of the halo profile.

        Parameters
        ----------
        cluster_weak_lensing : ClusterWeakLensing
            Cluster weak lensing object. Its statistics modeling (tables and
            selection function), profile (overdensity, source redshift
            distribution) and halo concentration are used for the covariance.
        n_eff : float or numpy.ndarray
            Effective source number density (gal/arcmin^2) of the source sample
            starting at the tomographic bin t, i.e. of all the galaxies in the
            tomographic bins t, t+1, ..., including those in front of the
            clusters. Either a float, used for all the source samples, or an
            array with one value per tomographic bin t of the profile,
            decreasing with t. A single value is a good approximation only for
            the lowest observed redshift bins.
        sigma_e : float
            Shape noise per shear component.
        nonlinear_perturbations : Perturbations, None
            Perturbations object providing the non-linear matter power spectrum,
            used for C_ell^kappakappa. If None, the linear matter power spectrum
            of the matter statistics perturbations is used.
        source_delta_z : float
            Redshift buffer of the source selection: the source sample of a
            cluster at redshift z starts at the first tomographic bin with lower
            edge larger than z + source_delta_z (see
            `HaloProfileBase.get_tomo_bin_index`).
        source_selection : str
            Redshift at which the source sample of the clusters in an observed
            redshift bin is selected:

                * "z_true" : at the true redshift of each cluster, as in `get_gt`.
                  Clusters at different true redshifts of the same observed
                  redshift bin can have different source samples.
                * "z_obs_center" : at the center of the observed redshift bin
                  (single source sample).
        ell : numpy.ndarray
            Multipoles used in the ell integration (log-spaced recommended).
        z_los : numpy.ndarray, None
            Line-of-sight redshifts for the C_ell^kappakappa integral. If None,
            50 points linearly spaced in [1e-3, zs_max] are used.
        z_sources_size : int
            Number of source redshift points in [0, zs_max] for the lensing
            window function.
        k_nonlinear : numpy.ndarray
            Wavenumbers (h/Mpc) where the power spectrum used in C_ell^kappakappa
            is tabulated.

        Notes
        -----
        - The covariance is computed for gamma_t, and used for the reduced shear
          g_t under the weak shear approximation.
        - Radii are physical Mpc/h, converted to angles with the angular diameter
          distance at the mean true redshift of each (z_obs, lambda_obs) bin.
        - The 1-halo term of the halo-matter power spectrum uses a truncated NFW
          profile with the halo concentration of ``cluster_weak_lensing``.
        - Distances assume a flat universe, as in the volume element.
        - Source samples (see `source_selection`): with the Limber approximation,
          correlated pairs of clusters are at the same true redshift, hence have
          the same source sample, which enters the C_ell^hh terms inside the
          redshift integral. The lens shot-noise terms (same cluster) are
          averaged over the source samples of the clusters, and the intrinsic
          term couples the halos with the source samples of the other clusters.
          Clusters without tomographic source bins have zero signal (as in
          `get_gt`) and do not contribute to the covariance.
        """
        self.cluster_weak_lensing = cluster_weak_lensing
        self.n_eff = n_eff
        self.sigma_e = sigma_e
        self.nonlinear_perturbations = nonlinear_perturbations
        self.source_delta_z = source_delta_z
        if source_selection not in ("z_true", "z_obs_center"):
            raise ValueError(
                "source_selection must be 'z_true' or 'z_obs_center', "
                f"got {source_selection}."
            )
        self.source_selection = source_selection
        self.ell = ell
        self.z_los = (
            np.linspace(1.0e-3, self.profile.zs_max, 50) if z_los is None else z_los
        )
        self.z_sources_size = z_sources_size
        self.k_nonlinear = k_nonlinear

        # survey solid angle in steradians
        self.survey_solid_angle = (
            self.cluster_statitstics_modeling.area * (np.pi / 180.0) ** 2.0
        )
        # comoving mean densities (units: h^2 Msun / Mpc^3), total matter for the
        # lensing kernel and cdm + baryons (no neutrinos) for the halo model,
        # consistently with the halo mass function
        _rho_crit_0 = (
            derived_cosmology.rho_crit(self.background, 0.0) / self.background.h**2.0
        )
        self.mean_matter_density = (
            self.cluster_statitstics_modeling.halo_model_properties.Omega_m_0
            * _rho_crit_0
        )
        self.mean_cb_density = (
            self.cluster_statitstics_modeling.halo_model_properties.Omega_cb_0
            * _rho_crit_0
        )

    @property
    def cluster_statitstics_modeling(self):
        return self.cluster_weak_lensing.cluster_statitstics_modeling

    @property
    def profile(self):
        return self.cluster_weak_lensing.profile

    @property
    def background(self):
        return self.cluster_statitstics_modeling.halo_model_properties.background

    # -------------------
    # auxiliary functions
    # -------------------

    def _comoving_distance(self, z):
        """Comoving distance (units: Mpc/h)."""
        return self.background.comoving_distance(z) * self.background.h

    def _inverse_sigma_crit_comoving(self, z_lens, z_sources):
        r"""Inverse of the comoving critical surface mass density,
        :math:`(1+z_l)^2/\Sigma_{\rm crit}`, set to zero for z_sources <= z_lens.

        Parameters
        ----------
        z_lens : numpy.ndarray
            Lens redshifts, shape (z_lens,).
        z_sources : numpy.ndarray
            Source redshifts, shape (z_sources,).

        Returns
        -------
        numpy.ndarray
            Inverse comoving critical surface mass density (units: Mpc^2 / h / Msun).
            Shape: (z_lens, z_sources).
        """
        fact = (units.SPEED_OF_LIGHT / 1.0e3 / units.MPC_TO_KM) ** 2.0 / (
            4.0 * np.pi * units.GRAVITATIONAL_CONSTANT
        )  # Msun/Mpc
        chi_l = self._comoving_distance(z_lens)[:, np.newaxis]
        chi_s = self._comoving_distance(z_sources)[np.newaxis, :]
        inv_sig_crit = (
            chi_l * (chi_s - chi_l) / chi_s * (1.0 + z_lens[:, np.newaxis]) / fact
        )
        return np.where(
            z_sources[np.newaxis, :] > z_lens[:, np.newaxis], inv_sig_crit, 0.0
        )

    def _source_windows(self, z_lens):
        r"""Lensing window function of the source sample starting at each
        tomographic bin t (Wu et al. 2019, Eq. 7):

        ..math:
            F_\kappa^t(z) = \bar\rho_m \int_0^{z_{s,\max}} dz_s\, p_t(z_s)\,
            \Sigma_{\rm crit, com}^{-1}(z, z_s)

        with p_t(z_s) the redshift distribution of the galaxies in the
        tomographic bins t, t+1, ..., normalized in [0, zs_max] as in
        `HaloProfileBase.sigma_crit_inv_eff` (the sources in front of the lens
        do not contribute).

        Parameters
        ----------
        z_lens : numpy.ndarray
            Redshifts where the window is evaluated.

        Returns
        -------
        F_kappa : numpy.ndarray
            Lensing window function (units: h / Mpc). Shape: (tomo_bins, z_lens).
        n_sources : numpy.ndarray
            Source surface number density (units: sr^-1). Shape: (tomo_bins,).
        """
        n_tomo = self.profile.z_min_tomo_bin.size
        z_s = np.linspace(0.0, self.profile.zs_max, self.z_sources_size)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_sig_crit = self._inverse_sigma_crit_comoving(z_lens, z_s)
        F_kappa = np.array(
            [
                self.mean_matter_density
                * self.profile.n_zs_norM(idx)
                * simpson(self.profile.n_zs(z_s, idx) * inv_sig_crit, x=z_s, axis=1)
                for idx in range(n_tomo)
            ]
        )
        # gal/arcmin^2 -> gal/sr
        n_sources = (
            np.broadcast_to(self.n_eff, (n_tomo,)) * (180.0 * 60.0 / np.pi) ** 2.0
        )
        return F_kappa, n_sources

    def _source_assignment(self, z_obs_edges):
        r"""Tomographic bin t where the source sample of the clusters at each
        true redshift of an observed redshift bin starts (see `source_selection`).

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins.

        Returns
        -------
        numpy.ndarray
            Index of the first tomographic bin, -1 if there are no source bins.
            Shape: (z_obs, ztrue).
        """
        z_true = self.cluster_statitstics_modeling.tabulated_integrands["ztrue"]
        if self.source_selection == "z_true":
            z_sel = np.tile(z_true, (len(z_obs_edges) - 1, 1))
        else:
            z_bin = 0.5 * (z_obs_edges[:-1] + z_obs_edges[1:])
            z_sel = np.repeat(z_bin[:, np.newaxis], z_true.size, axis=1)
        tomo_bins = [
            self.profile.get_tomo_bin_index(z, self.source_delta_z)
            for z in z_sel.flatten()
        ]
        return np.array([-1 if t is None else t for t in tomo_bins]).reshape(
            z_sel.shape
        )

    @staticmethod
    def _limber_power_spectrum(pk_table, k_table, z_chi, ell):
        r"""Evaluate a tabulated power spectrum at the Limber wavenumber
        :math:`k = (\ell + 1/2)/\chi(z)`, log-interpolating in k and setting it to
        zero outside the tabulated k range.

        Parameters
        ----------
        pk_table : numpy.ndarray
            Power spectrum tabulated at (z, k_table) (units: (Mpc/h)^3).
        k_table : numpy.ndarray
            Wavenumbers of the table (units: h/Mpc).
        z_chi : numpy.ndarray
            Comoving distance at each redshift of the table (units: Mpc/h).
        ell : numpy.ndarray
            Multipoles.

        Returns
        -------
        numpy.ndarray
            Power spectrum at the Limber wavenumber. Shape: (z, ell).
        """
        log_k = np.log((ell[np.newaxis, :] + 0.5) / z_chi[:, np.newaxis])
        log_k_table = np.log(k_table)
        return np.array(
            [
                np.exp(np.interp(_log_k, log_k_table, np.log(_pk)))
                * (_log_k >= log_k_table[0])
                * (_log_k <= log_k_table[-1])
                for _log_k, _pk in zip(log_k, pk_table)
            ]
        )

    @staticmethod
    def _nfw_fourier_transform(k, rs, c):
        r"""Normalized Fourier transform of the truncated NFW profile, u(k|M) -> 1
        for k -> 0 (Cooray & Sheth 2002, Eq. 81).

        Parameters
        ----------
        k : numpy.ndarray
            Wavenumbers (units: h/Mpc).
        rs : numpy.ndarray
            Comoving scale radius (units: Mpc/h), broadcastable with k.
        c : float
            Concentration.

        Returns
        -------
        numpy.ndarray
            u(k|M), dimensionless.
        """
        x = k * rs
        si_x, ci_x = sici(x)
        si_cx, ci_cx = sici((1.0 + c) * x)
        m_nfw = np.log(1.0 + c) - c / (1.0 + c)
        return (
            np.sin(x) * (si_cx - si_x)
            - np.sin(c * x) / ((1.0 + c) * x)
            + np.cos(x) * (ci_cx - ci_x)
        ) / m_nfw

    @staticmethod
    def _bin_averaged_J2(ell, theta_edges):
        r"""Bessel function J_2 averaged in annuli (Wu et al. 2019, Eq. B4):

        ..math:
            \hat J_2 = \frac{2}{\ell^2(\theta_{\max}^2-\theta_{\min}^2)}
            \left[2\left(J_0(\ell\theta_{\min})-J_0(\ell\theta_{\max})\right)
            + \ell\left(\theta_{\min}J_1(\ell\theta_{\min})
            - \theta_{\max}J_1(\ell\theta_{\max})\right)\right]

        Parameters
        ----------
        ell : numpy.ndarray
            Multipoles.
        theta_edges : numpy.ndarray
            Edges of angular bins (radians), shape (..., theta_edges).

        Returns
        -------
        numpy.ndarray
            Bin averaged J_2. Shape: (..., theta, ell).
        """
        theta_min = theta_edges[..., :-1, np.newaxis]
        theta_max = theta_edges[..., 1:, np.newaxis]
        return (
            2.0
            * (
                2.0 * (j0(ell * theta_min) - j0(ell * theta_max))
                + ell
                * (theta_min * j1(ell * theta_min) - theta_max * j1(ell * theta_max))
            )
            / (ell**2.0 * (theta_max**2.0 - theta_min**2.0))
        )

    # ------------------------
    # angular power spectra
    # ------------------------

    def _lens_sample(
        self, z_obs_edges, lambda_obs_edges, intermediate_integration_products=None
    ):
        r"""Lens sample quantities predicted from the halo mass function.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins.
        intermediate_integration_products : dict, None
            Intermediate products of ClusterWeakLensing.get_gt or get_DeltaSigma
            computed with the same redshift and richness bins. If provided, the
            window function, number density and cluster counts are taken from it
            instead of being recomputed.

        Returns
        -------
        dict
            Contains :

                * window_redshift_lambda_obs (numpy.ndarray) : P(lambda_obs_bin, z_obs_bin|M, ztrue), (z_obs, lambda_obs, ztrue, M).
                * cluster_counts (numpy.ndarray) : Number counts N, (z_obs, lambda_obs).
                * dN_dz (numpy.ndarray) : Normalized true redshift distribution p(ztrue) = dN/dz / N, (z_obs, lambda_obs, ztrue).
                * bias_dN_dz (numpy.ndarray) : b_eff(ztrue) * p(ztrue), (z_obs, lambda_obs, ztrue).
                * z_eff (numpy.ndarray) : Mean true redshift, (z_obs, lambda_obs).
        """
        csm = self.cluster_statitstics_modeling
        dv_dz = csm.tabulated_integrands["dv/dz(ztrue)"]

        if intermediate_integration_products is None:
            # P(lambda_obs_bin,z_obs_bin|M, z) : (z_obs, lambda_obs, ztrue, M)
            window_redshift_lambda_obs = csm.window_redshift_richness_observed(
                z_obs_edges, lambda_obs_edges
            )
            # comoving number density of the lens sample : (z_obs, lambda_obs, ztrue)
            number_density = csm.integrate_probe_function_in_mass(
                np.ones((1, 1)), window_redshift_lambda_obs
            )
            # cluster counts : (z_obs, lambda_obs)
            cluster_counts = csm.integrate_probe_function_in_redshift(number_density)
        else:
            window_redshift_lambda_obs = intermediate_integration_products[
                "window_redshift_lambda_obs"
            ]
            number_density = intermediate_integration_products["number_density"]
            cluster_counts = intermediate_integration_products["cluster_counts"]
            expected_shape = (len(z_obs_edges) - 1, len(lambda_obs_edges) - 1)
            if cluster_counts.shape != expected_shape:
                raise ValueError(
                    f"The intermediate integration products have {cluster_counts.shape} "
                    f"(z_obs, lambda_obs) bins, expected {expected_shape}."
                )
        # bias weighted comoving number density : (z_obs, lambda_obs, ztrue)
        bias_number_density = csm.integrate_probe_function_in_mass(
            csm.tabulated_integrands["bias(ztrue,M)"], window_redshift_lambda_obs
        )

        _norm = np.divide(
            dv_dz,
            cluster_counts[:, :, np.newaxis],
            out=np.zeros(number_density.shape),
            where=cluster_counts[:, :, np.newaxis] != 0,
        )
        dN_dz = number_density * _norm
        return {
            "window_redshift_lambda_obs": window_redshift_lambda_obs,
            "cluster_counts": cluster_counts,
            "dN_dz": dN_dz,
            "bias_dN_dz": bias_number_density * _norm,
            "z_eff": simpson(
                dN_dz * csm.tabulated_integrands["ztrue"],
                x=csm.tabulated_integrands["ztrue"],
                axis=2,
            ),
        }

    def _cl_hh(self, lens_sample, pk_lin_limber, factor):
        r"""Halo-halo angular power spectrum (Wu et al. 2019, Eq. 5), with
        :math:`P_{hh} = b_{\rm eff}^i b_{\rm eff}^j P_{\rm lin}`, multiplied by
        a redshift-dependent factor X (source term of the clusters at z):

        ..math:
            \int dz \frac{p_i(z)b_i(z)\,p_j(z)b_j(z)}{dV/dzd\Omega}
            P_{\rm lin}\left(\frac{\ell+1/2}{\chi(z)}, z\right) X(z, \ell)

        Parameters
        ----------
        lens_sample : dict
            Output of `_lens_sample`.
        pk_lin_limber : numpy.ndarray
            Linear power spectrum at the Limber wavenumber, shape (ztrue, ell).
        factor : numpy.ndarray
            X, shape (z_obs, ztrue, ell) or (z_obs, ztrue, 1).

        Returns
        -------
        numpy.ndarray
            Shape: (z_obs, lambda_obs, lambda_obs, ell).
        """
        csm = self.cluster_statitstics_modeling
        z_true = csm.tabulated_integrands["ztrue"]
        dV_dzdO = derived_cosmology.dV_dzdO(self.background, z_true, hubble_units=True)
        bias_dN_dz = lens_sample["bias_dN_dz"]
        return simpson(
            (
                bias_dN_dz[:, :, np.newaxis, :]
                * bias_dN_dz[:, np.newaxis, :, :]
                / dV_dzdO
            )[..., np.newaxis]
            * pk_lin_limber
            * factor[:, np.newaxis, np.newaxis, :, :],
            x=z_true,
            axis=3,
        )

    def _cl_hkappa(self, lens_sample, pk_lin_limber, F_kappa_ztrue, assignment):
        r"""Halo-convergence angular power spectrum (Wu et al. 2019, Eq. 8), with
        the halo-matter power spectrum given by the 1-halo (NFW) and 2-halo terms,
        for the halos with source sample s and the convergence of the source
        sample t:

        ..math:
            C_\ell^{h_i^s\kappa_t} = \int dz\, p_i(z) \delta_{s, s(z)}
            \frac{F_\kappa^t(z)}{\chi^2(z)}
            \left[\frac{1}{n_i(z)}\int dM \frac{dn}{dM} W_i(M,z)\frac{M}{\bar\rho_{cb}}u(k|M,z)
            + b_i(z) P_{\rm lin}(k, z)\right]_{k=(\ell+1/2)/\chi(z)}

        Parameters
        ----------
        lens_sample : dict
            Output of `_lens_sample`.
        pk_lin_limber : numpy.ndarray
            Linear power spectrum at the Limber wavenumber, shape (ztrue, ell).
        F_kappa_ztrue : numpy.ndarray
            Lensing window of the source samples, shape (tomo_bins, ztrue).
        assignment : numpy.ndarray
            Output of `_source_assignment`, shape (z_obs, ztrue).

        Returns
        -------
        numpy.ndarray
            Zero for the source samples not used. Shape: (z_obs, lambda_obs,
            tomo_bins s, tomo_bins t, ell).
        """
        csm = self.cluster_statitstics_modeling
        z_true = csm.tabulated_integrands["ztrue"]
        mass = csm.tabulated_integrands["M"]
        chi = self._comoving_distance(z_true)
        window = lens_sample["window_redshift_lambda_obs"]

        # comoving NFW scale radius : (ztrue, M, 1)
        _, RDelta, _ = self.profile.surface_mass_density_args(np.zeros(1), z_true, mass)
        rs = (
            RDelta
            * (1.0 + z_true[:, np.newaxis, np.newaxis])
            / self.cluster_weak_lensing.halo_concentration
        )
        # M/rho_cb u(k|M) : (ztrue, M, ell)
        k_limber = (self.ell[np.newaxis, :] + 0.5) / chi[:, np.newaxis]
        mass_profile = (
            mass[np.newaxis, :, np.newaxis]
            / self.mean_cb_density
            * self._nfw_fourier_transform(
                k_limber[:, np.newaxis, :],
                rs,
                self.cluster_weak_lensing.halo_concentration,
            )
        )

        # 1-halo term times p_i(z) : (z_obs, lambda_obs, ztrue, ell)
        _norm = np.divide(
            csm.tabulated_integrands["dv/dz(ztrue)"],
            lens_sample["cluster_counts"][:, :, np.newaxis],
            out=np.zeros(lens_sample["dN_dz"].shape),
            where=lens_sample["cluster_counts"][:, :, np.newaxis] != 0,
        )
        p_hm_1h = (
            np.concatenate(
                [
                    csm.integrate_probe_function_in_mass(
                        mass_profile, window[ind_z : ind_z + 1, :, :, :, np.newaxis]
                    )
                    for ind_z in range(window.shape[0])
                ]
            )
            * _norm[..., np.newaxis]
        )
        # 2-halo term times p_i(z) : (z_obs, lambda_obs, ztrue, ell)
        p_hm_2h = lens_sample["bias_dN_dz"][..., np.newaxis] * pk_lin_limber

        p_hm = (p_hm_1h + p_hm_2h) / chi[:, np.newaxis] ** 2.0

        # source samples of the clusters with nonzero density
        n_tomo = F_kappa_ztrue.shape[0]
        _has_clusters = np.any(lens_sample["dN_dz"] > 0.0, axis=1)
        used = [t for t in range(n_tomo) if np.any((assignment == t) & _has_clusters)]
        cl_hk = np.zeros(p_hm.shape[:2] + (n_tomo, n_tomo, self.ell.size))
        for s in used:
            p_hm_s = p_hm * (assignment == s)[:, np.newaxis, :, np.newaxis]
            for t in used:
                cl_hk[:, :, s, t] = simpson(
                    F_kappa_ztrue[t][:, np.newaxis] * p_hm_s, x=z_true, axis=2
                )
        return cl_hk

    def _cl_kappakappa(self, F_kappa_los):
        r"""Convergence angular power spectrum (Wu et al. 2019, Eq. 6):

        ..math:
            C_\ell^{\kappa\kappa} = \int dz \frac{d\chi}{dz}\frac{F_\kappa^2(z)}{\chi^2(z)}
            P_{\rm NL}\left(\frac{\ell+1/2}{\chi(z)}, z\right)

        Returns
        -------
        numpy.ndarray
            Shape: (F_kappa_los.shape[0], ell).
        """
        h = self.background.h
        chi = self._comoving_distance(self.z_los)
        dchi_dz = (
            units.SPEED_OF_LIGHT
            / 1.0e3
            / self.background.hubble_parameter(self.z_los)
            * h
        )

        perturbations = (
            self.cluster_statitstics_modeling.halo_model_properties.perturbations
            if self.nonlinear_perturbations is None
            else self.nonlinear_perturbations
        )
        pk_table = perturbations.matter_power_spectrum(
            self.z_los, self.k_nonlinear, hubble_units=True, k_hunit=True
        )
        pk_limber = self._limber_power_spectrum(
            pk_table, self.k_nonlinear, chi, self.ell
        )

        return simpson(
            (dchi_dz * F_kappa_los**2.0 / chi**2.0)[:, :, np.newaxis] * pk_limber,
            x=self.z_los,
            axis=1,
        )

    # ----------------------
    # covariance
    # ----------------------

    def get_gt_covariance(
        self,
        z_obs_edges,
        lambda_obs_edges,
        radius_edges,
        cross_richness=True,
        return_terms=False,
        intermediate_integration_products=None,
    ):
        """Compute the Gaussian covariance of the reduced shear profile.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        radius_edges : numpy.ndarray
            Edges of radial bins for the profile (units: physical Mpc/h).
        cross_richness : bool
            If True, computes also the covariance between different richness
            bins in the same redshift bin, else only the auto richness blocks.
        return_terms : bool
            If True, returns also the shape noise, LSS and intrinsic terms
            (Wu et al. 2019, Eqs. 11-13), which sum to the covariance.
        intermediate_integration_products : dict, None
            Intermediate products returned by ClusterWeakLensing.get_gt (or
            get_DeltaSigma) with ``return_intermediate_products=True``, computed
            with the same redshift and richness bins. If provided, the window
            function, number density and cluster counts are not recomputed.

        Returns
        -------
        cov_gt : numpy.ndarray
            Covariance of the reduced shear in redshift, richness and radial bins,
            diagonal in redshift bins, zero in the redshift bins without
            tomographic source bins.
            Dimension: (z_obs, z_obs, lambda_obs, lambda_obs, radius, radius).
        terms (optional) : dict
            Dictionary with the "shape", "lss" and "intr" terms, same dimension
            as cov_gt. Returned only when `return_terms` is true.
        """
        csm = self.cluster_statitstics_modeling
        z_true = csm.tabulated_integrands["ztrue"]
        z_obs_edges_size = len(z_obs_edges) - 1
        lambda_obs_edges_size = len(lambda_obs_edges) - 1
        radius_edges_size = len(radius_edges) - 1

        ##############################
        # Lens and source sample
        ##############################
        lens_sample = self._lens_sample(
            z_obs_edges, lambda_obs_edges, intermediate_integration_products
        )
        # lens surface density in sr^-1 : (z_obs, lambda_obs)
        n_lens = lens_sample["cluster_counts"] / self.survey_solid_angle

        # lensing window functions and number density of the source samples
        # starting at each tomographic bin : (tomo, ztrue), (tomo, z_los), (tomo,)
        F_kappa_ztrue, n_sources = self._source_windows(z_true)
        F_kappa_los, _ = self._source_windows(self.z_los)
        # source sample of the clusters at each true redshift : (z_obs, ztrue)
        assignment = self._source_assignment(z_obs_edges)
        # masks of the clusters with source sample t : (tomo, z_obs, ztrue)
        masks = np.array([assignment == t for t in range(n_sources.size)], dtype=float)
        # fraction of the clusters of each bin with source sample t : (z_obs, lambda_obs, tomo)
        source_fractions = simpson(
            lens_sample["dN_dz"][:, :, np.newaxis, :]
            * masks.transpose(1, 0, 2)[:, np.newaxis, :, :],
            x=z_true,
            axis=3,
        )

        ##############################
        # Angular power spectra
        ##############################
        chi_true = self._comoving_distance(z_true)
        k_table = csm.tabulated_integrands["k"]
        pk_lin_limber = self._limber_power_spectrum(
            csm.halo_model_properties.matter_power_spectrum_cb(z_true, k_table),
            k_table,
            chi_true,
            self.ell,
        )
        # convergence of each source sample : (tomo, ell)
        cl_kk = self._cl_kappakappa(F_kappa_los)
        # shape noise of each source sample : (tomo,)
        shape_noise_sources = self.sigma_e**2.0 / n_sources

        # pairs of clusters (same true redshift, i.e. same source sample) :
        # (z_obs, lambda_obs, lambda_obs, ell)
        cl_hh_shape = self._cl_hh(
            lens_sample,
            pk_lin_limber,
            np.einsum("tzx,t->zx", masks, shape_noise_sources)[..., np.newaxis],
        )
        cl_hh_kk = self._cl_hh(
            lens_sample, pk_lin_limber, np.einsum("tzx,tl->zxl", masks, cl_kk)
        )
        # halos with source sample s and convergence of the source sample t :
        # (z_obs, lambda_obs, tomo s, tomo t, ell)
        cl_hk = self._cl_hkappa(lens_sample, pk_lin_limber, F_kappa_ztrue, assignment)

        if not cross_richness:
            _identity = np.identity(lambda_obs_edges_size)[np.newaxis, :, :, np.newaxis]
            cl_hh_shape *= _identity
            cl_hh_kk *= _identity

        # lens shot noise : (z_obs, lambda_obs, lambda_obs)
        shot_noise_lens = np.divide(
            np.identity(lambda_obs_edges_size)[np.newaxis, :, :],
            n_lens[:, :, np.newaxis],
            out=np.zeros(
                (z_obs_edges_size, lambda_obs_edges_size, lambda_obs_edges_size)
            ),
            where=n_lens[:, :, np.newaxis] != 0,
        )
        # same cluster, averaged over the source samples of the clusters :
        # (z_obs, lambda_obs, 1, 1, 1), (z_obs, lambda_obs, ell)
        shape_noise_auto = (source_fractions @ shape_noise_sources)[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ]
        cl_kk_auto = source_fractions @ cl_kk

        # integrands in ell : (z_obs, lambda_obs, lambda_obs, ell)
        # the lens shot noise x shape noise term is computed analytically below
        _integrands = {
            "shape": cl_hh_shape,
            "lss": cl_hh_kk
            + shot_noise_lens[..., np.newaxis] * cl_kk_auto[:, :, np.newaxis, :],
            "intr": np.einsum("zistl,zjtsl->zijl", cl_hk, cl_hk),
        }
        if not cross_richness:
            _integrands["intr"] *= np.identity(lambda_obs_edges_size)[
                np.newaxis, :, :, np.newaxis
            ]

        ##############################
        # Hankel transform in radial bins
        ##############################
        # radius to angle at the mean true redshift : (z_obs, lambda_obs, radius_edges)
        D_A = (
            self.background.angular_diameter_distance(lens_sample["z_eff"].flatten())
            * self.background.h
        ).reshape(lens_sample["z_eff"].shape)
        theta_edges = radius_edges[np.newaxis, np.newaxis, :] / D_A[:, :, np.newaxis]
        # (z_obs, lambda_obs, radius, ell)
        hat_J2 = self._bin_averaged_J2(self.ell, theta_edges)

        # ell d ell / 2pi -> ell^2 / 2pi dln(ell), trapezoidal weights in ln(ell)
        _dlnell = np.diff(np.log(self.ell))
        _weights = np.zeros(self.ell.size)
        _weights[:-1] += 0.5 * _dlnell
        _weights[1:] += 0.5 * _dlnell
        _weights *= self.ell**2.0 / (2.0 * np.pi)

        fsky = self.survey_solid_angle / (4.0 * np.pi)
        terms = {
            name: np.einsum(
                "ziaL,zjbL,zijL,L->zijab", hat_J2, hat_J2, integrand, _weights
            )
            / (4.0 * np.pi * fsky)
            for name, integrand in _integrands.items()
        }

        # lens shot noise x shape noise, using int ell dell/2pi J2_a J2_b = delta_ab/A_a
        # (Wu et al. 2019, App. B): sigma_e^2 / (N_h n_s A_a)
        annulus_area = np.pi * (
            theta_edges[..., 1:] ** 2.0 - theta_edges[..., :-1] ** 2.0
        )
        terms["shape"] += (
            shot_noise_lens[..., np.newaxis, np.newaxis]
            * shape_noise_auto
            / (4.0 * np.pi * fsky)
            * (np.identity(radius_edges_size) / annulus_area[:, :, np.newaxis, :])[
                :, np.newaxis, :, :, :
            ]
        )

        # make it (z_obs, z_obs, lambda_obs, lambda_obs, radius, radius), diagonal in z_obs
        _z_identity = np.identity(z_obs_edges_size)[
            :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]
        terms = {
            name: _z_identity * term[np.newaxis, ...] for name, term in terms.items()
        }
        cov_gt = terms["shape"] + terms["lss"] + terms["intr"]

        if return_terms:
            return cov_gt, terms
        return cov_gt
