from copy import copy
from typing import Tuple
import jax.numpy as jnp
import pdb


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

    a_t = jnp.sum(1 / a) ** (-1)

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

    * Pg_result -> the optimal dispatch by merit order
    * lambd -> the marginal cost considering the optimal dispatch

    """

    # Initialize the result
    Pg_result = jnp.zeros_like(Pg_max)

    # Check if the generators are able to generate the demand
    if Pg_max.sum().item() < demand:
        raise ValueError("Demand greater than generators' maximum capacity")
    elif demand < Pg_min.sum().item():
        raise ValueError("Demand smaller than generators' minimum capacity")

    # Calculate the cost parameters of the equivalent machine
    a_t = calculate_a_t(a)
    b_t = calculate_b_t(a, b, a_t)

    # Calculate the incremental cost for the given demand
    lambd = calculate_equivalent_incremental_cost(demand, a_t, b_t)

    # Calculate the initial dispatch
    Pg = calculate_dispatched_power_from_cost(lambd, a, b)

    original_demand = copy(demand)

    while Pg_result.sum().item() != original_demand:

        lower_violation = Pg - Pg_min
        lower_violation = jnp.where(lower_violation > 0.0, 0.0, lower_violation)
        lower_violation = jnp.abs(lower_violation)

        upper_violation = Pg - Pg_max
        upper_violation = jnp.where(upper_violation < 0.0, 0.0, upper_violation)
        upper_violation = jnp.abs(upper_violation)

        if (
            jnp.count_nonzero(lower_violation) == 0
            and jnp.count_nonzero(upper_violation) == 0
        ):
            Pg_result = jnp.where(Pg_result == 0.0, Pg, Pg_result)
            break

        # Check which generator violated the most
        index_max_lower_violation = jnp.argmax(lower_violation)
        max_lower_violation = lower_violation[index_max_lower_violation]

        index_max_upper_violation = jnp.argmax(upper_violation)
        max_upper_violation = upper_violation[index_max_upper_violation]

        if max_lower_violation > max_upper_violation:
            # Fix the violation
            fixed_dispatch = Pg_min[index_max_lower_violation].copy()
            Pg_result = Pg_result.at[index_max_lower_violation].set(fixed_dispatch)
        else:
            # Fix the violation
            fixed_dispatch = Pg_max[index_max_upper_violation].copy()
            Pg_result = Pg_result.at[index_max_upper_violation].set(fixed_dispatch)

        demand = demand - fixed_dispatch

        non_dispatched_generators = jnp.argwhere(Pg_result == 0.0)

        # Update the parameters
        a_t = calculate_a_t(a[non_dispatched_generators])
        b_t = calculate_b_t(
            a[non_dispatched_generators], b[non_dispatched_generators], a_t
        )
        lambd = calculate_equivalent_incremental_cost(demand, a_t, b_t)

        # Redispatch
        Pg = calculate_dispatched_power_from_cost(lambd, a, b)

        # Fix the already dispatched generators
        Pg = jnp.where(Pg_result != 0.0, Pg_result, Pg)

    return Pg_result, lambd
