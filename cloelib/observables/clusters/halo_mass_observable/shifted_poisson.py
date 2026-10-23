# General imports
# import jax.numpy as np
import numpy as np
import scipy.special as spc

from cloelib.observables.clusters.auxiliary import tabulated_return

_HALF_LN_2PI = 0.9189385332046727


def _shifted_poisson_logpdf(lambda_true, lsat, s2, x_switch=10.0):
    r"""
    Logarithm of the shifted-Poisson richness PDF, numerically stable.

    The direct expression

    .. math::
        \ln P = -\nu + (x - 1)\ln\nu - \ln\Gamma(x), \quad
        \nu = l_{\rm sat} + s^2, \quad x = \lambda_{\rm true} + s^2, \quad
        s = \sigma_{\rm lnltr}\, l_{\rm sat}

    For :math:`x > x_{\rm switch}` the Stirling series of :math:`\ln\Gamma(x)`
    is used:

    .. math::
        \ln P = (x - 1)\,{\rm log1p}\!\left(\frac{l_{\rm sat} - \lambda_{\rm true}}{x}\right)
        + (\lambda_{\rm true} - l_{\rm sat}) - \tfrac{1}{2}\ln x - \tfrac{1}{2}\ln 2\pi
        - \frac{1}{12x} + \frac{1}{360x^3} + O(x^{-5}).

    For :math:`x \le x_{\rm switch}` the direct expression is used. With :math:`x_{\rm switch} = 10` the truncation error
    of the series is below :math:`10^{-7}` in :math:`\ln P`.

    Parameters
    ----------
    lambda_true: numpy.ndarray
        True richness points.
    lsat: numpy.ndarray
        Mean number of satellites, :math:`\langle\lambda_{\rm sat}\rangle`.
    s2: numpy.ndarray
        Square of the super-Poisson scatter, :math:`(\sigma_{\rm lnltr} l_{\rm sat})^2`.
    x_switch: float
        Value of :math:`x` above which the stable (Stirling) form is used.

    Returns
    -------
    logpdf: numpy.ndarray
        :math:`\ln P(\lambda_{\rm true} | z, M)`, broadcast over the inputs.
    """
    nu = lsat + s2
    x = lambda_true + s2
    direct = -nu + (x - 1.0) * np.log(nu) - spc.gammaln(x)
    xs = np.maximum(x, x_switch)
    inv = 1.0 / xs
    stable = (
        (xs - 1.0) * np.log1p((lsat - lambda_true) * inv)
        + (lambda_true - lsat)
        - 0.5 * np.log(xs)
        - _HALF_LN_2PI
        - inv * (1.0 / 12.0 - inv * inv / 360.0)
    )
    return np.where(x > x_switch, stable, direct)


class ShiftedPoissonHaloMassObservable:
    def __init__(
        self,
        Mmin: float,
        M1: float,
        alpha: float,
        epsilon: float,
        sigma_lnltr: float,
        z_piv: float = 1.25,
    ):
        r"""
        Class defining the selection function of galaxy clusters, including
        sample purity, completeness, mass-observable relation, and
        uncertainties on observed quantities.

        Parameters
        ----------
        Mmin : float
            Minimum halo mass to host a central galaxy
        M1 : float
            Minimum halo mass to host one satellite galaxy
        alpha : float
            Mass slope of the proxy - mass scaling relation
        epsilon : float
            Redshift evolution of the poxy-mass scaling relation
        z_piv: float
            Redshift pivot in the proxy - mass relation
        """
        self.Mmin = Mmin
        self.M1 = M1
        self.alpha = alpha
        self.epsilon = epsilon
        self.sigma_lnltr = sigma_lnltr
        self.z_piv = z_piv

        # to avoid recomputing pdf_richness
        self._tabulated_pdf_richness = {
            "inputs": {
                "M": None,
                "z": None,
                "lambda_true": None,
            },
            "values": None,
        }

    def _mean_richness(self, z, M):
        r"""
        Mean of the richness-mass relation PDF.

        Computes the theoretical richness at
        the requested true redshift and mass points.

        .. math::
            \lambda_{\rm true}(z, M) = 1 + \left(\frac{M - M_{\text{min}}}{M_1 - M_{\text{min}}}\right)^{\alpha} 
            \left(\frac{1 + z}{1 + z_{\text{piv}}}\right)^{\epsilon}

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.

        Returns
        -------
        lnrichness : numpy.ndarray
            ln(richness), with the same dimensions of the
            operation z x M.
        """

        lsat = (((M-self.Mmin)/(self.M1-self.Mmin))**self.alpha
                *((1.+z)/(1.+self.z_piv))**self.epsilon)
        return (1. + lsat)

    def scatter_richness(self, z, M):
        r"""
        Intrinsic scatter of the proxy - mass relation.

        Computes the scatter of the theoretical richness probability distribution
        at the requested true redshift and mass points.

        .. math::
            \text{scatter} = \sigma_{\text{lnltr}} \cdot l_{\text{sat}} =
            \sigma_{\text{lnltr}} \cdot (\text{\_mean\_lnrichness}(z, M) - 1)

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.

        Returns
        -------
        scatter_lnrichness : numpy.ndarray
            Scatter of ln(richness), with the same dimensions of the
            operation z x M.
        """

        return self.sigma_lnltr*(self._mean_richness(z,M)-1.)
        

    def _pdf_richness(self, z, M, lambda_true):
        r"""
        Proxy - mass relation PDF.

        Computes the theoretical richness probability distribution
        at the requested true mass, redshift, and richness points.

        Continuous Poisson shift-corrected by the variance.
        .. math::
            P(\lambda_{\text{true}} \mid z, M) = \frac{e^{-\Lambda} \cdot
            \Lambda^{X - 1}}{\Gamma(X)}

        The logarithm is evaluated with :func:`_shifted_poisson_logpdf`,
        which avoids the catastrophic cancellation of the direct expression
        for large scatter (accurate also in single precision).

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        pdf_richness: numpy.ndarray
            PDF of richness.
        """

        lsat = self._mean_richness(z, M) - 1.0
        s2 = self.scatter_richness(z, M) ** 2.0
        logpdf = _shifted_poisson_logpdf(lambda_true, lsat, s2)
        return np.exp(logpdf, dtype="float128")

    def pdf_richness(self, z, M, lambda_true):
        r"""
        Proxy - mass relation PDF.

        Computes the theoretical richness probability distribution
        at the requested true mass, redshift, and richness points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.
        lambda_true: numpy.ndarray
            True richness points.

        Returns
        -------
        pdf_richness: numpy.ndarray
            pdf_richness[i,j,k], where i is the redshift, j is the mass,
            and k is the true richness index
        """
        return tabulated_return(
            self._tabulated_pdf_richness,
            self._pdf_richness,
            {
                "z": np.atleast_1d(z)[:, np.newaxis, np.newaxis],
                "M": np.atleast_1d(M)[np.newaxis, :, np.newaxis],
                "lambda_true": np.atleast_1d(lambda_true)[np.newaxis, np.newaxis, :],
            },
        )

