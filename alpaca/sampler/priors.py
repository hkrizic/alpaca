"""Prior distribution constructors for sampler parameters."""

from scipy.stats import truncnorm
from scipy.stats import uniform as uniform_dist


def tnorm(mu, sigma, lo, hi):
    """Construct a truncated normal distribution.

    Parameterizes scipy.stats.truncnorm using physical bounds rather than
    standardized clip values, simplifying prior specification for bounded
    lens model parameters.

    Args:
        mu: Distribution mean (mode for symmetric truncation).
        sigma: Standard deviation before truncation.
        lo: Lower bound defining the support.
        hi: Upper bound defining the support.

    Returns:
        Frozen truncated normal distribution (scipy.stats.truncnorm).
    """
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma)


def _bounded_prior(
    mu,
    sigma,
    lo,
    hi,
    use_uniform_for_bounded: bool = False,
    uniform_widen_factor: float = 1.0,
):
    """Construct a bounded prior distribution.

    Returns either a truncated normal centered at mu or a uniform distribution
    over a possibly widened interval. Used internally for sampling priors
    on physically constrained parameters.

    Args:
        mu: Central value for truncated normal.
        sigma: Scale parameter for truncated normal.
        lo: Lower bound on parameter support.
        hi: Upper bound on parameter support.
        use_uniform_for_bounded: If True, return uniform instead of truncated normal.
        uniform_widen_factor: Factor to expand the uniform support interval.

    Returns:
        scipy.stats distribution (truncnorm or uniform).
    """
    if use_uniform_for_bounded:
        center = 0.5 * (lo + hi)
        half_width = 0.5 * (hi - lo) * float(uniform_widen_factor)
        lo_u = center - half_width
        hi_u = center + half_width
        return uniform_dist(loc=lo_u, scale=hi_u - lo_u)
    else:
        return tnorm(mu, sigma, lo, hi)
