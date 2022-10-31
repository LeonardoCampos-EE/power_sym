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
    demand = 250.0

    Pg, lambd = merit_order_dispatch_algorithm(Pg_max, Pg_min, a, b, demand)

    exact_dispatch = np.array([100.0, 150.0])
    exact_lambd = 7.84

    np.testing.assert_allclose(exact_dispatch, np.array(Pg), rtol=1e-1)
    np.testing.assert_approx_equal(exact_lambd, lambd, significant=1)

    return
