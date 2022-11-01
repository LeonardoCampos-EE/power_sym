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

    * Pg -> the optimal dispatch by merit order
    * lambd -> the marginal cost considering the optimal dispatch

    """

    # Calculate the number of generation units
    num_units = len(Pg_max)

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

    # Initialize the result
    Pg_result = jnp.zeros_like(Pg_max)

    # Calculate the total generation
    Pd = Pg.sum().item()

    # Calculate the difference between generation and demand
    delta_P = abs(Pd - demand)

    while Pd != demand:

        lower_violation = Pg - Pg_min
        lower_violation = jnp.where(lower_violation > 0.0, 0.0, lower_violation)
        lower_violation = jnp.abs(lower_violation)

        upper_violation = Pg - Pg_max
        upper_violation = jnp.where(upper_violation < 0.0, 0.0, upper_violation)
        upper_violation = jnp.abs(upper_violation)

        # Check which generator violated the most
        index_max_lower_violation = jnp.argmax(lower_violation)
        max_lower_violation = lower_violation[index_max_lower_violation]

        index_max_upper_violation = jnp.argmax(upper_violation)
        max_upper_violation = upper_violation[index_max_upper_violation]

        if max_lower_violation > max_upper_violation:
            # Fix the violation
            Pg_result[index_max_lower_violation] = Pg_min[
                index_max_lower_violation
            ].copy()
        else:
            # Fix the violation
            Pg_result[index_max_upper_violation] = Pg_min[
                index_max_upper_violation
            ].copy()

    # Iterate through the generators to check if their dispatch is within
    # the given limits, fix it otherwise
    for idx in range(num_units):

        # Get the dispatch of the current generator
        dispatch = (lambd - b[idx]) / a[idx]
        Pg_current = Pg_current.at[idx].set(dispatch)

        # Check for a violation on the lower bound
        if dispatch < Pg_min[idx]:
            # Fix the dispatch
            fixed_dispatch = Pg_min[idx]
            Pg_current = Pg_current.at[idx].set(fixed_dispatch)

            # Re-calculate the cost paramters
            if idx < num_units - 1:
                a_t = calculate_a_t(a[idx + 1 :])
                b_t = calculate_b_t(a[idx + 1 :], b[idx + 1 :], a_t)
            else:
                a_t = calculate_a_t(a[idx])
                b_t = calculate_b_t(a[idx], b[idx], a_t)

            # Remove the current generator's power output from the demand
            demand = demand - fixed_dispatch

            # Update the marginal cost
            lambd = calculate_equivalent_incremental_cost(demand, a_t, b_t)

        elif dispatch > Pg_max[idx]:

            # Fix the dispatch
            fixed_dispatch = Pg_max[idx]
            Pg_current = Pg_current.at[idx].set(fixed_dispatch)

            # Re-calculate the cost paramters
            if idx < num_units - 1:
                a_t = calculate_a_t(a[idx + 1 :])
                b_t = calculate_b_t(a[idx + 1 :], b[idx + 1 :], a_t)
            else:
                a_t = calculate_a_t(a[idx])
                b_t = calculate_b_t(a[idx], b[idx], a_t)

            # Remove the current generator's power output from the demand
            demand = demand - fixed_dispatch

            # Update the marginal cost
            lambd = calculate_equivalent_incremental_cost(demand, a_t, b_t)

    return Pg_current, lambd
