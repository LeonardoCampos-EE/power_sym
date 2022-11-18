from functools import partial
from typing import Callable
import jax.numpy as jnp
from jax import grad, jacfwd
import jax

jax.config.update("jax_platform_name", "cpu")
import numpy as np
import pdb

from methods import log_barrier, restrictions, newton_method, optimize, fun

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

    def res_for_test(x, extra_variables=None):
        return -x[0] - x[1]

    actual_barrier = log_barrier(X1, res_for_test)
    expected_barrier = -0.6931472
    np.testing.assert_approx_equal(actual_barrier, expected_barrier, significant=1)

    return


def test_newton_method() -> None:
    @jax.jit
    def f(x, t=0.0, extra_variables=None):
        return (x[0] * x[1] - 2) ** 2 + (x[1] - 3) ** 2

    H = jax.jacfwd(jax.jacrev(f, argnums=0), argnums=0)
    J = jax.jacfwd(f, argnums=0)

    x0 = jnp.asarray([1.0, 2.0])

    x = newton_method(x0, f, J, H)
    x = np.asarray(x)

    expected_x = np.array([2.0 / 3.0, 3.0])

    np.testing.assert_allclose(x, expected_x, atol=1e-3)
