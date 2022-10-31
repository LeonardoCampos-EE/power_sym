from typing import Tuple
import jax.numpy as jnp


def generator_cost_function(
    Pg: jnp.DeviceArray, a: jnp.DeviceArray, b: jnp.DeviceArray, c: jnp.DeviceArray
) -> jnp.DeviceArray:

    cost = jnp.multiply(a / 2, Pg**2) + jnp.multiply(b, Pg) + c

    return cost


def incremental_cost_function(
    Pg: jnp.DeviceArray, a: jnp.DeviceArray, b: jnp.DeviceArray
) -> jnp.DeviceArray:

    incremental_cost = jnp.multiply(a / 2, Pg) + b

    return incremental_cost


def calculate_a_t(a: jnp.DeviceArray) -> float:

    a_t = 1.0 / jnp.sum(1 / a)

    return a_t.item()


def calculate_b_t(a: jnp.DeviceArray, b: jnp.DeviceArray, a_t: float) -> float:

    b_t = a_t * jnp.sum(b / a)

    return b_t.item()


def calculate_Pg_t(Pg: jnp.DeviceArray) -> float:

    return jnp.sum(Pg).item()


def calculate_equivalent_incremental_cost(Pg_t: float, a_t: float, b_t: float) -> float:

    equivalent_incremental_cost = a_t * Pg_t + b_t

    return equivalent_incremental_cost


def calculate_dispatched_power_from_cost(
    equivalent_incremental_cost: float, a: jnp.DeviceArray, b: jnp.DeviceArray
) -> jnp.DeviceArray:

    Pg_dispatched = (equivalent_incremental_cost - b) / a

    return Pg_dispatched


def calculate_dispatched_power_from_demand(
    demand: float, a: jnp.DeviceArray, b: jnp.DeviceArray
) -> jnp.DeviceArray:

    a_t = 1.0 / jnp.sum(1 / a)
    b_t = a_t * jnp.sum(b / a)

    equivalent_incremental_cost = a_t.item() * demand + b_t.item()
    Pg_dispatched = (equivalent_incremental_cost - b) / a

    return Pg_dispatched


def merit_order_dispatch_algorithm(
    Pg_max: jnp.DeviceArray,
    Pg_min: jnp.DeviceArray,
    a: jnp.DeviceArray,
    b: jnp.DeviceArray,
    demand: float,
) -> Tuple[jnp.DeviceArray, float]:

    """
    Returns a tuple containing:

    * Pg -> the optimal dispatch by merit order
    * lambd -> the marginal cost considering the optimal dispatch

    """

    # Calculate the cost parameters of the equivalent machine
    a_t = calculate_a_t(a)
    b_t = calculate_b_t(a, b, a_t)

    # Calculate the incremental cost for the given demand
    lambd = calculate_equivalent_incremental_cost(demand, a_t, b_t)

    # Calculate the initial dispatch
    Pg_current = calculate_dispatched_power_from_cost(lambd, a, b)

    # Iterate through the generators to check if their dispatch is within
    # the given limits, fix it otherwise
    for idx in range(len(Pg_current)):

        # Get the dispatch of the current generator
        dispatch = Pg_current[idx]

        # Check for a violation on the lower bound
        if dispatch < Pg_min[idx]:
            # Fix the dispatch
            fixed_dispatch = Pg_min[idx]

            # Remove the current generator from the calculations
            a = a.at[idx].set(jnp.inf)
            b = b.at[idx].set(0.0)

            # Re-calculate the cost paramters
            a_t = calculate_a_t(a)
            b_t = calculate_b_t(a, b, a_t)

            # Remove the current generator's power output from the demand
            demand = demand - fixed_dispatch

            # Update the marginal cost
            lambd = calculate_equivalent_incremental_cost(demand, a_t, b_t)

            # Re-calculate the dispatches
            Pg_current = calculate_dispatched_power_from_cost(lambd, a, b)

            # Update the current generator's dispatch
            Pg_current = Pg_current.at[idx].set(fixed_dispatch)

        elif dispatch > Pg_max[idx]:

            # Fix the dispatch
            fixed_dispatch = Pg_max[idx]

            # Remove the current generator from the calculations
            a = a.at[idx].set(jnp.inf)
            b = b.at[idx].set(0.0)

            # Re-calculate the cost paramters
            a_t = calculate_a_t(a)
            b_t = calculate_b_t(a, b, a_t)

            # Remove the current generator's power output from the demand
            demand = demand - fixed_dispatch

            # Update the marginal cost
            lambd = calculate_equivalent_incremental_cost(demand, a_t, b_t)

            # Re-calculate the dispatches
            Pg_current = calculate_dispatched_power_from_cost(lambd, a, b)

            # Update the current generator's dispatch
            Pg_current = Pg_current.at[idx].set(fixed_dispatch)

    return Pg_current, lambd
