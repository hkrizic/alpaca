"""JAX utility functions for pytree manipulation."""

import jax
from jax.tree_util import tree_leaves


@jax.jit
def get_value_from_index(xs, i):
    """Extract the i-th element from each leaf of a pytree.

    Args:
        xs: JAX pytree with array leaves.
        i: Index to extract.

    Returns:
        Pytree with same structure, each leaf replaced by its i-th element.
    """
    return jax.tree.map(lambda x: x[i], xs)


def is_batched_pytree(params):
    """Check if a pytree contains batched (multi-sample) parameters.

    Args:
        params: JAX pytree to check.

    Returns:
        True if all leaves are arrays with shape[0] > 1.
    """
    leaves = tree_leaves(params)
    return all(hasattr(x, 'shape') and x.ndim > 0 and x.shape[0] > 1 for x in leaves)
