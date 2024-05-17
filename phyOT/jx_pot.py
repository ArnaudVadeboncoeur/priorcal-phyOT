import jax
import jax.numpy as jnp
from jax import random

def zero_pad(a, pad_width, value=0):
    return jnp.pad(a, pad_width, constant_values=value)

def searchsorted(a, v, side='left'):
    if a.ndim == 1:
        return jnp.searchsorted(a, v, side)
    else:
        # this is a not very efficient way to make jax numpy
        # searchsorted work on 2d arrays
        # loop_way = jnp.array([jnp.searchsorted(a[i, :], v[i, :], side) for i in range(a.shape[0])])
        # this is an efficient way to do it.
        vmap_way = jax.vmap(jnp.searchsorted, in_axes=(0,0,None))(a, v, side)
        return vmap_way



def quantile_function(qs, cws, xs):
    r""" Computes the quantile function of an empirical distribution

    Parameters
    ----------
    qs: array-like, shape (n,)
        Quantiles at which the quantile function is evaluated
    cws: array-like, shape (m, ...)
        cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
    xs: array-like, shape (n, ...)
        locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

    Returns
    -------
    q: array-like, shape (..., n)
        The quantiles of the distribution
    """

    n = xs.shape[0]

    cws = cws.T
    qs = qs.T
    idx = searchsorted(cws, qs).T
    return jnp.take_along_axis(xs, jnp.clip(idx, 0, n - 1), axis=0)

def wasserstein_1d(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    r"""
    Computes the 1 dimensional OT loss [15] between two (batched) empirical
    distributions

    .. math:
        OT_{loss} = \int_0^1 |cdf_u^{-1}(q) - cdf_v^{-1}(q)|^p dq

    It is formally the p-Wasserstein distance raised to the power p.
    We do so in a vectorized way by first building the individual quantile functions then integrating them.

    This function should be preferred to `emd_1d` whenever the backend is
    different to numpy, and when gradients over
    either sample positions or weights are required.

    Parameters
    ----------
    u_values: array-like, shape (n, ...)
        locations of the first empirical distribution
    v_values: array-like, shape (m, ...)
        locations of the second empirical distribution
    u_weights: array-like, shape (n, ...), optional
        weights of the first empirical distribution, if None then uniform weights are used
    v_weights: array-like, shape (m, ...), optional
        weights of the second empirical distribution, if None then uniform weights are used
    p: int, optional
        order of the ground metric used, should be at least 1 (see [2, Chap. 2], default is 1
    require_sort: bool, optional
        sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
        the function, default is True

    Returns
    -------
    cost: float/array-like, shape (...)
        the batched EMD

    References
    ----------
    .. [15] Peyré, G., & Cuturi, M. (2018). Computational Optimal Transport.

    """

    assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

    n = u_values.shape[0]
    m = v_values.shape[0]

    if u_weights is None:
        u_weights = jnp.full(u_values.shape, 1. / n, dtype=u_values.dtype)
    elif u_weights.ndim != u_values.ndim:
        u_weights = jnp.repeat(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = jnp.full(v_values.shape, 1. / m, dtype=v_values.dtype)
    elif v_weights.ndim != v_values.ndim:
        v_weights = jnp.repeat(v_weights[..., None], v_values.shape[-1], -1)

    if require_sort:
        u_sorter = jnp.argsort(u_values, 0)
        u_values = jnp.take_along_axis(u_values, u_sorter, 0)

        v_sorter = jnp.argsort(v_values, 0)
        v_values = jnp.take_along_axis(v_values, v_sorter, 0)

        u_weights = jnp.take_along_axis(u_weights, u_sorter, 0)
        v_weights = jnp.take_along_axis(v_weights, v_sorter, 0)

    u_cumweights = jnp.cumsum(u_weights, 0)
    v_cumweights = jnp.cumsum(v_weights, 0)

    qs = jnp.sort(jnp.concatenate((u_cumweights, v_cumweights), 0), 0)
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)
    qs = zero_pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = jnp.abs(u_quantiles - v_quantiles)

    if p == 1:
        return jnp.sum(delta * diff_quantiles, axis=0)
    return jnp.sum(delta * jnp.power(diff_quantiles, p), axis=0)



def get_random_projections(key, d, n_projections):
    r"""
    Generates n_projections samples from the uniform on the unit sphere of dimension :math:`d-1`: :math:`\mathcal{U}(\mathcal{S}^{d-1})`

    Parameters
    ----------
    d : int
        dimension of the space
    n_projections : int
        number of samples requested
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    backend:
        Backend to ue for random generation

    Returns
    -------
    out: ndarray, shape (d, n_projections)
        The uniform unit vectors on the sphere

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_random_projections(d, n_projections)
    >>> np.allclose(np.sum(np.square(projs), 0), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True

    """
    projections = random.normal(key, shape=(d, n_projections))
    projections = projections / jnp.sqrt(jnp.sum(projections**2, 0, keepdims=True))
    return projections



def sliced_wasserstein_distance(key, X_s, X_t, a=None, b=None, n_projections=50, p=2,
                                projections=None, seed=None, log=False):
    r"""
    Computes a Monte-Carlo approximation of the p-Sliced Wasserstein distance

    .. math::
        \mathcal{SWD}_p(\mu, \nu) = \underset{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}{\mathbb{E}}\left(\mathcal{W}_p^p(\theta_\# \mu, \theta_\# \nu)\right)^{\frac{1}{p}}


    where :

    - :math:`\theta_\# \mu` stands for the pushforwards of the projection :math:`X \in \mathbb{R}^d \mapsto \langle \theta, X \rangle`


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional =
        Power p used for computing the sliced Wasserstein
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45
    """

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = jnp.full(n, 1 / n, dtype=X_s.dtype)
    if b is None:
        b = jnp.full(m, 1 / m, dtype=X_s.dtype)

    d = X_s.shape[1]

    if projections is None:
        projections = get_random_projections(key, d, n_projections)
    else:
        n_projections = projections.shape[1]

    X_s_projections = jnp.dot(X_s, projections)
    X_t_projections = jnp.dot(X_t, projections)

    projected_emd = wasserstein_1d(X_s_projections, X_t_projections, a, b, p=p)

    res = (jnp.sum(projected_emd) / n_projections) ** (1.0 / p)
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res



def max_sliced_wasserstein_distance(key, X_s, X_t, a=None, b=None, n_projections=50, p=2,
                                    projections=None, log=False):
    r"""
    Computes a Monte-Carlo approximation of the max p-Sliced Wasserstein distance

    .. math::
        \mathcal{Max-SWD}_p(\mu, \nu) = \underset{\theta _in
        \mathcal{U}(\mathbb{S}^{d-1})}{\max} [\mathcal{W}_p^p(\theta_\#
        \mu, \theta_\# \nu)]^{\frac{1}{p}}

    where :

    - :math:`\theta_\# \mu` stands for the pushforwards of the projection :math:`\mathbb{R}^d \ni X \mapsto \langle \theta, X \rangle`


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional =
        Power p used for computing the sliced Wasserstein
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [35] Deshpande, I., Hu, Y. T., Sun, R., Pyrros, A., Siddiqui, N., Koyejo, S., ... & Schwing, A. G. (2019). Max-sliced wasserstein distance and its use for gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10648-10656).
    """

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = jnp.full(n, 1 / n, dtype=X_s.dtype)
    if b is None:
        b = jnp.full(m, 1 / m, dtype=X_s.dtype)

    d = X_s.shape[1]

    if projections is None:
        projections = get_random_projections(key, d, n_projections)

    X_s_projections = jnp.dot(X_s, projections)
    X_t_projections = jnp.dot(X_t, projections)

    projected_emd = wasserstein_1d(X_s_projections, X_t_projections, a, b, p=p)

    res = jnp.max(projected_emd) ** (1.0 / p)
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res



def sliced_wasserstein_distance_CDiag(key, X_s, X_t, C, n_projections=50, p=2.):
    r"""
    Computes a Monte-Carlo approximation of the p-Sliced Wasserstein distance

    .. math::
        \mathcal{SWD}_p(\mu, \nu) = \underset{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}{\mathbb{E}}\left(\mathcal{W}_p^p(\theta_\# \mu, \theta_\# \nu)\right)^{\frac{1}{p}}


    where :

    - :math:`\theta_\# \mu` stands for the pushforwards of the projection :math:`X \in \mathbb{R}^d \mapsto \langle \theta, X \rangle`


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional =
        Power p used for computing the sliced Wasserstein
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45
    """

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))
    a = jnp.full(n, 1 / n, dtype=X_s.dtype)
    b = jnp.full(m, 1 / m, dtype=X_s.dtype)

    d = X_s.shape[1]

    projections = random.normal(key, shape=(d, n_projections))
    projections = projections / jnp.sqrt(jnp.sum(projections**2, 0, keepdims=True))
    C_min_half_projections = jnp.einsum('i, ib->ib', C**(-0.5), projections )
    X_s_projections = jnp.dot(X_s, C_min_half_projections)
    X_t_projections = jnp.dot(X_t, C_min_half_projections)

    projected_emd = wasserstein_1d(X_s_projections, X_t_projections, a, b, p=p)

    res = (jnp.sum(projected_emd) / n_projections) ** (1.0 / p)

    return res