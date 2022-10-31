import jax.numpy as jnp
import numpy as np
import pdb

from power_flow.merit_order_dispatch import (
    generator_cost_function,
    incremental_cost_function,
    equivalent_incremental_cost_function,
    calculate_dispatched_power_from_cost,
    calculate_dispatched_power_from_demand,
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


def test_equivalent_incremental_cost_function() -> None:

    Pg = jnp.array([45.4545, 204.5455])

    exact_incremental_cost = 8.3636

    estimated_cost = equivalent_incremental_cost_function(Pg, a, b)

    np.testing.assert_approx_equal(
        exact_incremental_cost, estimated_cost, significant=1
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
