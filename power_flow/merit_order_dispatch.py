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


def equivalent_incremental_cost_function(
    Pg: jnp.DeviceArray, a: jnp.DeviceArray, b: jnp.DeviceArray
) -> float:

    a_t = 1.0 / jnp.sum(1 / a)
    Pg_t = jnp.sum(Pg)
    b_t = a_t * jnp.sum(b / a)

    equivalent_incremental_cost = a_t * Pg_t + b_t

    return equivalent_incremental_cost.item()


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
