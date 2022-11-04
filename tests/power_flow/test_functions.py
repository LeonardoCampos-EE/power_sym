import jax.numpy as jnp
import numpy as np
import pdb
from power_flow.functions import (
    generator_cost_function,
    forward_active_power_flow,
    backward_active_power_flow,
    active_power_flow,
)


def test_generator_cost_function() -> None:

    P_g = jnp.array([2.0, 2.0, 2.0])
    a = jnp.array([2.0, 2.0, 2.0])
    b = jnp.array([3.0, 3.0, 3.0])
    c = jnp.array([1.0, 1.0, 1.0])

    exact_cost = 45.0

    calculated_cost = generator_cost_function(P_g, a, b, c)

    np.testing.assert_approx_equal(exact_cost, calculated_cost)

    return


def test_forward_active_power_flow() -> None:

    V = jnp.array([1.050, 0.9547, 1.0])
    theta = jnp.array([0.0, -2.767, 0.694])
    G = jnp.array([[0.0, 1.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    B = jnp.array([[0.0, -2.0, -2.0], [0.0, 0.0, -4.0], [0.0, 0.0, 0.0]])

    expected_P_km = np.array([[0.0, 0.198, 0.027], [0.0, 0.0, -0.313], [0.0, 0.0, 0.0]])
    P_km = forward_active_power_flow(V, theta, G, B)
    P_km = np.asarray(P_km)

    np.testing.assert_allclose(P_km, expected_P_km, rtol=1e-1)


def test_backward_active_power_flow() -> None:

    V = jnp.array([1.050, 0.9547, 1.0])
    theta = jnp.array([0.0, -2.767, 0.694])
    G = jnp.array([[0.0, 1.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    B = jnp.array([[0.0, -2.0, -2.0], [0.0, 0.0, -4.0], [0.0, 0.0, 0.0]])

    expected_P_mk = np.array(
        [[0.0, 0.0, 0.0], [-0.187, 0.0, 0.0], [-0.025, 0.325, 0.0]]
    )
    P_mk = backward_active_power_flow(V, theta, G, B)
    P_mk = np.asarray(P_mk)

    np.testing.assert_allclose(P_mk, expected_P_mk, rtol=1e-1)


def test_active_power_flow() -> None:

    V = jnp.array([1.050, 0.9547, 1.0])
    theta = jnp.array([0.0, -2.767, 0.694])
    G = jnp.array([[0.0, 1.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    B = jnp.array([[0.0, -2.0, -2.0], [0.0, 0.0, -4.0], [0.0, 0.0, 0.0]])

    expected_P = np.array(
        [[0.0, 0.198, 0.027], [-0.187, 0.0, -0.313], [-0.025, 0.325, 0.0]]
    )
    P = active_power_flow(V, theta, G, B)
    P = np.asarray(P)

    np.testing.assert_allclose(P, expected_P, rtol=1e-1)
