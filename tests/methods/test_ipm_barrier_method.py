import jax.numpy as jnp
from jax import grad
import numpy as np
import pdb

from methods import log_barrier, restrictions

X = jnp.array([[5.0], [6.0]])


def test_log_barrier() -> None:

    g = grad(log_barrier, argnums=0)

    expected_grad = jnp.array(
        [
            [
                2 * (X[0] - 5)
                - (
                    -2 * X[0] / (-X[0] ** 2 + 4)
                    + jnp.exp(-X[0]) / (-jnp.exp(-X[0]) + X[1])
                    + (-1) / (-X[0] - 2 * X[1] + 4)
                    + 1 / X[0]
                )
            ],
            [
                2 * (X[1] - 6)
                - (
                    1 / (-jnp.exp(-X[0]) + X[1])
                    + (-2) / (-X[0] - 2 * X[1] + 4)
                    + 1 / X[1]
                )
            ],
        ]
    )[:, 0, :]
    actual_grad = g(X, restrictions)

    expected_grad = np.asarray(expected_grad)
    actual_grad = np.asarray(actual_grad)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-1)

    X1 = jnp.array([[1.0], [1.0]])
    actual_barrier = log_barrier(X1, lambda x: -x[0] - x[1])
    expected_barrier = -0.6931472
    np.testing.assert_approx_equal(actual_barrier, expected_barrier, significant=1)

    return
