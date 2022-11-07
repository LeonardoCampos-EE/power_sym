import jax.numpy as jnp
import numpy as np
import pdb

from methods import fun, restrictions

X = jnp.array([[5.0], [6.0]])


def test_fun() -> None:

    value_expected = 0.0
    value = fun(X)

    np.testing.assert_approx_equal(value, value_expected, significant=1)

    return


def test_restrictions() -> None:

    expected_res = np.array([[21.0], [-5.993262053000914], [13.0], [5.0], [6.0]])
    res = restrictions(X)

    res = np.asarray(res)

    np.testing.assert_allclose(res, expected_res, rtol=1e-1)
