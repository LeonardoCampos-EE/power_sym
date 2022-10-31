import jax.numpy as jnp
import numpy as np

from power_flow import generator_cost_function


def test_generator_cost_function() -> None:

    P_g = jnp.array([2.0, 2.0, 2.0])
    a = jnp.array([2.0, 2.0, 2.0])
    b = jnp.array([3.0, 3.0, 3.0])
    c = jnp.array([1.0, 1.0, 1.0])

    exact_cost = 45.0

    calculated_cost = generator_cost_function(P_g, a, b, c)

    np.testing.assert_approx_equal(exact_cost, calculated_cost)

    return
    