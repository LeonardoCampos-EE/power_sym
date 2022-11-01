import jax.numpy as jnp
import numpy as np
import pdb

from power_flow.merit_order_dispatch import (
    generator_cost_function,
    incremental_cost_function,
    calculate_a_t,
    calculate_b_t,
    calculate_Pg_t,
    calculate_equivalent_incremental_cost,
    calculate_dispatched_power_from_cost,
    calculate_dispatched_power_from_demand,
    merit_order_dispatch_algorithm,
)

# Tests conducted in the 2 generator example system
a = jnp.array([0.008, 0.0096])
b = jnp.array([8.0, 6.4])
c = jnp.array([0.0, 0.0])


def test_generator_cost_function() -> None:

    Pg = jnp.array([45.4545, 204.5455])

    exact_cost = 1881.80

    estimated_cost = generator_cost_function(Pg, a, b, c).sum().item()

    np.testing.assert_approx_equal(exact_cost, estimated_cost, significant=1)

    return


def test_incremental_cost_function() -> None:

    Pg = jnp.array([1000.0, 1000.0])
    exact_cost = np.array([12.0, 11.2])

    estimated_cost = np.array(incremental_cost_function(Pg, a, b))
    np.testing.assert_allclose(estimated_cost, exact_cost)

    return


def test_calculate_equivalent_incremental_cost() -> None:

    Pg_t = 250.0

    a_t = calculate_a_t(a)

    np.testing.assert_approx_equal(a_t, 0.004363636486232281)

    b_t = calculate_b_t(a, b, a_t)

    np.testing.assert_approx_equal(b_t, 7.2727274894714355)

    exact_incremental_cost = 8.3636

    estimated_cost = calculate_equivalent_incremental_cost(Pg_t, a_t, b_t)

    np.testing.assert_approx_equal(
        estimated_cost, exact_incremental_cost, significant=1
    )

    return


def test_calculate_dispatched_power_from_cost() -> None:

    exact_dispatch = np.array([45.4545, 204.5455])
    cost = 8.3636

    estimated_dispatch = np.array(calculate_dispatched_power_from_cost(cost, a, b))

    np.testing.assert_allclose(exact_dispatch, estimated_dispatch, rtol=1e-1, atol=1e-1)

    return


def test_calculate_dispatched_power_from_demand() -> None:

    exact_dispatch = np.array([45.4545, 204.5455])
    demand = 250.0

    estimated_dispatch = np.array(calculate_dispatched_power_from_demand(demand, a, b))

    np.testing.assert_allclose(exact_dispatch, estimated_dispatch, rtol=1e-1, atol=1e-1)

    return


def test_merit_order_dispatch_algoritm() -> None:

    Pg_max = jnp.array([625.0, 625.0])
    Pg_min = jnp.array([100.0, 100.0])

    # Case 1
    demand = 250.0
    exact_dispatch = np.array([100.0, 150.0])
    exact_lambd = 7.84

    Pg, lambd = merit_order_dispatch_algorithm(Pg_max, Pg_min, a, b, demand)

    np.testing.assert_allclose(exact_dispatch, np.array(Pg), rtol=1e-1)
    np.testing.assert_approx_equal(exact_lambd, lambd, significant=1)
    np.testing.assert_approx_equal(demand, jnp.sum(Pg).item(), significant=1)

    # Case 2
    demand = 350.0
    exact_dispatch = np.array([100.0, 250.0])
    exact_lambd = 8.8

    Pg, lambd = merit_order_dispatch_algorithm(Pg_max, Pg_min, a, b, demand)

    np.testing.assert_allclose(exact_dispatch, np.array(Pg), rtol=1e-1)
    np.testing.assert_approx_equal(exact_lambd, lambd, significant=1)
    np.testing.assert_approx_equal(demand, jnp.sum(Pg).item(), significant=1)

    # Case 3
    demand = 500.0
    exact_dispatch = np.array([181.82, 318.18])
    exact_lambd = 9.45

    Pg, lambd = merit_order_dispatch_algorithm(Pg_max, Pg_min, a, b, demand)

    np.testing.assert_allclose(exact_dispatch, np.array(Pg), rtol=1e-1)
    np.testing.assert_approx_equal(exact_lambd, lambd, significant=1)
    np.testing.assert_approx_equal(demand, jnp.sum(Pg).item(), significant=1)

    # Case 4
    demand = 700
    exact_dispatch = np.array([290.91, 409.09])
    exact_lambd = 10.33

    Pg, lambd = merit_order_dispatch_algorithm(Pg_max, Pg_min, a, b, demand)

    np.testing.assert_allclose(exact_dispatch, np.array(Pg), rtol=1e-1)
    np.testing.assert_approx_equal(exact_lambd, lambd, significant=1)
    np.testing.assert_approx_equal(demand, jnp.sum(Pg).item(), significant=1)

    # Case 5
    demand = 900.0
    exact_dispatch = np.array([400.0, 500.0])
    exact_lambd = 11.2

    Pg, lambd = merit_order_dispatch_algorithm(Pg_max, Pg_min, a, b, demand)

    np.testing.assert_allclose(exact_dispatch, np.array(Pg), rtol=1e-1)
    np.testing.assert_approx_equal(exact_lambd, lambd, significant=1)
    np.testing.assert_approx_equal(demand, jnp.sum(Pg).item(), significant=1)

    # Case 6
    demand = 1100.0
    exact_dispatch = np.array([509.09, 590.91])
    exact_lambd = 12.07

    Pg, lambd = merit_order_dispatch_algorithm(Pg_max, Pg_min, a, b, demand)

    np.testing.assert_allclose(exact_dispatch, np.array(Pg), rtol=1e-1)
    np.testing.assert_approx_equal(exact_lambd, lambd, significant=1)
    np.testing.assert_approx_equal(demand, jnp.sum(Pg).item(), significant=1)

    # Case 7
    demand = 1175.0
    exact_dispatch = np.array([550.0, 625.0])
    exact_lambd = 12.4

    Pg, lambd = merit_order_dispatch_algorithm(Pg_max, Pg_min, a, b, demand)

    np.testing.assert_allclose(exact_dispatch, np.array(Pg), rtol=1e-1)
    np.testing.assert_approx_equal(exact_lambd, lambd, significant=1)
    np.testing.assert_approx_equal(demand, jnp.sum(Pg).item(), significant=1)

    # Case 8
    demand = 1250.0
    exact_dispatch = np.array([625.0, 625.0])
    exact_lambd = 13.0

    Pg, lambd = merit_order_dispatch_algorithm(Pg_max, Pg_min, a, b, demand)

    np.testing.assert_allclose(exact_dispatch, np.array(Pg), rtol=1e-1)
    np.testing.assert_approx_equal(exact_lambd, lambd, significant=1)
    np.testing.assert_approx_equal(demand, jnp.sum(Pg).item(), significant=1)

    return
