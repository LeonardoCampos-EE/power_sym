import jax.numpy as jnp
from jax import grad
import numpy as np
import pdb
from power_flow.functions import (
    generator_cost_function,
    forward_active_power_flow,
    backward_active_power_flow,
    active_power_flow,
    forward_reactive_power_flow,
    backward_reactive_power_flow,
    reactive_power_flow,
    active_power_losses,
    active_power_balance,
    calculate_hydro_goal_deviation,
    objective_function,
    restrictions_function,
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

    return


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

    return


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

    return


def test_forward_reactive_power_flow() -> None:

    V = jnp.array([1.050, 0.9547, 1.0])
    theta = jnp.array([0.0, -2.767, 0.694])
    G = jnp.array([[0.0, 1.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    B = jnp.array([[0.0, -2.0, -2.0], [0.0, 0.0, -4.0], [0.0, 0.0, 0.0]])
    Bsh = jnp.array([[0.0, 0.01, 0.01], [0.0, 0.0, 0.01], [0.0, 0.0, 0.0]])

    expected_Qkm = np.array(
        [[0.0, 0.143, 0.1068], [0.0, 0.0, -0.0597], [0.0, 0.0, 0.0]]
    )
    Qkm = forward_reactive_power_flow(V, theta, G, B, Bsh)

    Qkm = np.asarray(Qkm)

    np.testing.assert_allclose(Qkm, expected_Qkm, rtol=1e-1)

    return


def test_backward_reactive_power_flow() -> None:

    V = jnp.array([1.050, 0.9547, 1.0])
    theta = jnp.array([0.0, -2.767, 0.694])
    G = jnp.array([[0.0, 1.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    B = jnp.array([[0.0, -2.0, -2.0], [0.0, 0.0, -4.0], [0.0, 0.0, 0.0]])
    Bsh = jnp.array([[0.0, 0.01, 0.01], [0.0, 0.0, 0.01], [0.0, 0.0, 0.0]])

    expected_Qmk = np.array(
        [[0.0, 0.0, 0.0], [-0.1403, 0.0, 0.0], [-0.1226, 0.0627, 0.0]]
    )
    Qmk = backward_reactive_power_flow(V, theta, G, B, Bsh)

    Qmk = np.asarray(Qmk)

    np.testing.assert_allclose(Qmk, expected_Qmk, rtol=1e-1)

    return


def test_reactive_power_flow() -> None:

    V = jnp.array([1.050, 0.9547, 1.0])
    theta = jnp.array([0.0, -2.767, 0.694])
    G = jnp.array([[0.0, 1.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    B = jnp.array([[0.0, -2.0, -2.0], [0.0, 0.0, -4.0], [0.0, 0.0, 0.0]])
    Bsh = jnp.array([[0.0, 0.01, 0.01], [0.0, 0.0, 0.01], [0.0, 0.0, 0.0]])

    expected_Q = np.array(
        [[0.0, 0.143, 0.1068], [-0.1403, 0.0, -0.0597], [-0.1226, 0.0627, 0.0]]
    )
    Q = reactive_power_flow(V, theta, G, B, Bsh)

    Q = np.asarray(Q)

    np.testing.assert_allclose(Q, expected_Q, rtol=1e-1)

    return


def test_power_losses() -> None:

    V = jnp.array([1.050, 0.9547, 1.0])
    theta = jnp.array([0.0, -2.767, 0.694])
    G = jnp.array([[0.0, 1.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    B = jnp.array([[0.0, -2.0, -2.0], [0.0, 0.0, -4.0], [0.0, 0.0, 0.0]])
    expected_Ploss = np.array(
        [[0.0, 0.0114, 0.0027], [0.0, 0.0, 0.0111], [0.0, 0.0, 0.0]]
    )

    P = active_power_flow(V, theta, G, B)
    Ploss = active_power_losses(P)
    Ploss = np.asarray(Ploss)

    np.testing.assert_allclose(Ploss, expected_Ploss, rtol=1e-1)

    return


def test_active_power_balance() -> None:

    P = jnp.array(
        [
            [0.0, 0.19801769, 0.02714109],
            [-0.18659827, 0.0, -0.3135509],
            [-0.02448714, 0.3246202, 0.0],
        ]
    )
    P_g = jnp.array([[0.22515878], [0.0], [0.30013305]])
    P_c = jnp.array([[0.0], [0.50014913], [0.0]])

    expected_delta_P = np.array([[0.0], [0.0], [0.0]])

    delta_P = active_power_balance(P, P_g, P_c)
    delta_P = np.asarray(delta_P)

    np.testing.assert_allclose(delta_P, expected_delta_P, rtol=1e-1)

    return


def test_reactive_power_balance() -> None:

    Q = jnp.array(
        [
            [0.0, 0.14305021, 0.10684675],
            [-0.14035083, 0.0, -0.05987231],
            [-0.12256378, 0.06289641, 0.0],
        ]
    )
    Q_g = jnp.array([[0.24989696], [0.0], [-0.05966737]])
    Q_c = jnp.array([[0.0], [0.20022315], [0.0]])

    expected_delta_Q = np.array([[0.0], [0.0], [0.0]])

    delta_Q = active_power_balance(Q, Q_g, Q_c)
    delta_Q = np.asarray(delta_Q)

    np.testing.assert_allclose(delta_Q, expected_delta_Q, rtol=1e-1)

    return


def test_calculate_hydro_goal_deviation() -> None:

    Pg = jnp.array(
        [
            [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ],
            [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
            ],
        ]
    )
    Pg = jnp.expand_dims(Pg, axis=1)

    assert Pg.shape == (2, 1, 24)

    goal = jnp.array([[2.4], [2.4]])
    assert goal.shape == (2, 1)

    expected_deviation = 2.4
    deviation = calculate_hydro_goal_deviation(Pg, goal)

    np.testing.assert_approx_equal(deviation, expected_deviation, significant=1)

    return


def test_objective_function() -> None:

    P_g = jnp.array([2.0, 2.0, 2.0] * 3).reshape((3, 1, 3))
    a = jnp.array([2.0, 2.0, 2.0] * 3).reshape((3, 1, 3))
    b = jnp.array([3.0, 3.0, 3.0] * 3).reshape((3, 1, 3))
    c = jnp.array([1.0, 1.0, 1.0] * 3).reshape((3, 1, 3))

    extra_variables = {"a": a, "b": b, "c": c}

    exact_cost = 45.0 * 3

    calculated_cost = objective_function(P_g, extra_variables)

    np.testing.assert_approx_equal(exact_cost, calculated_cost)

    return


def test_objective_function_gradient() -> None:

    P_g = jnp.array([2.0, 2.0, 2.0] * 3).reshape((3, 1, 3))
    a = jnp.array([2.0, 2.0, 2.0] * 3).reshape((3, 1, 3))
    b = jnp.array([3.0, 3.0, 3.0] * 3).reshape((3, 1, 3))
    c = jnp.array([1.0, 1.0, 1.0] * 3).reshape((3, 1, 3))

    extra_variables = {"a": a, "b": b, "c": c}

    exact_gradient = 11.0 * 9.0

    g = grad(objective_function, argnums=0)

    calculated_gradient = g(P_g, extra_variables).sum().item()

    np.testing.assert_approx_equal(exact_gradient, calculated_gradient)

    return
