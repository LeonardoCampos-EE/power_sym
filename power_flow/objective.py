import jax.numpy as jnp


def generator_cost_function(
    Pg: jnp.DeviceArray, a: jnp.DeviceArray, b: jnp.DeviceArray, c: jnp.DeviceArray
) -> float:

    cost = jnp.multiply(a, Pg**2) + jnp.multiply(b, Pg) + c
    cost = jnp.sum(cost).item()

    return cost




