import jax
import jax.numpy as jnp
import pdb


@jax.jit
def fun(x: jnp.DeviceArray) -> float:

    f = jnp.square(x[0] - 5) + jnp.square(x[1] - 6)
    f = f.astype(float)

    return f


@jax.jit
def restrictions(x: jnp.DeviceArray) -> jnp.DeviceArray:

    res_1 = jnp.square(x[0]) - 4
    res_2 = jnp.exp(-x[0]) - x[1]
    res_3 = x[0] + 2 * x[1] - 4
    res_4 = x[0]
    res_5 = x[1]
    res = jnp.array([res_1, res_2, res_3, res_4, res_5])

    return res
