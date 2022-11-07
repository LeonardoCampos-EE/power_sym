from typing import Callable
import jax
import jax.numpy as jnp
import pdb
from functools import partial


@partial(jax.jit, static_argnums=(1,))
def log_barrier(x: jnp.DeviceArray, restrictions: Callable) -> float:

    """
    This function calculates the logarithmic barrier for the
    Interior Point Method, which can be expressed as:

    P(x) = - sum( log(-h_i(x)) )

    In which h_i represents the i-th restriction of the problem and x
    represents its input.

    args:
        - x: a JAX array containing the input to the restrictions
        - restrictions: a Python function which evaluates the restrictions
        given an input x
    returns:
        - barrier: a float value calculated using the expression above
    """

    res = restrictions(x)
    barrier = jnp.log(-res)
    barrier = -jnp.sum(barrier)
    barrier = barrier.astype(float)

    return barrier
