from typing import Tuple
import jax
import jax.numpy as jnp
import pdb

# @jax.jit
def generator_cost_function(
    Pg: jnp.DeviceArray, a: jnp.DeviceArray, b: jnp.DeviceArray, c: jnp.DeviceArray
) -> float:

    cost = jnp.multiply(a, Pg**2) + jnp.multiply(b, Pg) + c
    cost = jnp.sum(cost).item()

    return cost


@jax.jit
def forward_active_power_flow(
    V: jnp.DeviceArray, theta: jnp.DeviceArray, G: jnp.DeviceArray, B: jnp.DeviceArray
) -> jnp.DeviceArray:

    """
    Computes the power flow from buses which start indexes are lower than
    end indexes, i.e., the P_km flow.

    Args:
        - V -> Jax ROW vector of N entries (N is the number of buses) containing the
        voltage magnitude of each bus
        - theta -> Jax ROW vector of N entries (N is the number of buses) containing the
        voltage angle of each bus (in degrees)
        - G -> Jax matrix of NxN entries, containing the conductance of each transmission
        line
        - B -> Jax matrix of NxN entries, containing the susceptance of each transmission
        line
    Returns:
        - P_km -> Jax matrix which the element ij is the active power flow from bus i
        to j

    The forward flow equation is

    P_km = Gkm V²_k - Gkm Vk Vm cos(theta_km) - Bkm Vk Vm sen(theta_km)

    Where k is the starting bus and m is the ending bus connected by a transmission line,
    theta_km = theta_k - theta_m

    An expression like theta_k - theta_m can be efficiently computed as:
    theta_km = theta - theta^T, by using Jax/Numpy broadcasting capabilities
    """

    # Auxiliary variables
    Vk = jnp.expand_dims(V, axis=0).T
    Vm = Vk.T

    theta_k = jnp.deg2rad(jnp.expand_dims(theta, axis=0)).T
    theta_m = theta_k.T

    # Operations
    theta_km = theta_k - theta_m
    VkVm = jnp.multiply(Vk, Vm)

    VkVm_cos_theta = jnp.multiply(VkVm, jnp.cos(theta_km))
    VkVm_sin_theta = jnp.multiply(VkVm, jnp.sin(theta_km))

    P_km = (
        jnp.multiply(G, jnp.square(Vk))
        - jnp.multiply(G, VkVm_cos_theta)
        - jnp.multiply(B, VkVm_sin_theta)
    )

    return P_km


@jax.jit
def backward_active_power_flow(
    V: jnp.DeviceArray, theta: jnp.DeviceArray, G: jnp.DeviceArray, B: jnp.DeviceArray
) -> jnp.DeviceArray:

    """
    Computes the power flow from buses which start indexes are greater than
    end indexes, i.e., the P_mk flow.

    Args:
        - V -> Jax ROW vector of N entries (N is the number of buses) containing the
        voltage magnitude of each bus
        - theta -> Jax ROW vector of N entries (N is the number of buses) containing the
        voltage angle of each bus (in degrees)
        - G -> Jax matrix of NxN entries, containing the conductance of each transmission
        line
        - B -> Jax matrix of NxN entries, containing the susceptance of each transmission
        line

    Returns
        - P_mk -> Jax matrix which the element ji is the active power flow from bus j
        to i

    The backward flow equation is

    P_mk = Gkm V²_m - Gkm Vk Vm cos(theta_km) + Bkm Vk Vm sen(theta_km)

    Where k is the starting bus and m is the ending bus connected by a transmission line,
    theta_km = theta_k - theta_m

    An expression like theta_k - theta_m can be efficiently computed as:
    theta_km = theta - theta^T, by using Jax/Numpy broadcasting capabilities
    """

    # Auxiliary variables
    Vk = jnp.expand_dims(V, axis=0).T
    Vm = Vk.T

    theta_k = jnp.deg2rad(jnp.expand_dims(theta, axis=0)).T
    theta_m = theta_k.T

    # Operations
    theta_km = theta_k - theta_m
    VkVm = jnp.multiply(Vk, Vm)

    VkVm_cos_theta = jnp.multiply(VkVm, jnp.cos(theta_km))
    VkVm_sin_theta = jnp.multiply(VkVm, jnp.sin(theta_km))

    P_mk = (
        jnp.multiply(G, jnp.square(Vm))
        - jnp.multiply(G, VkVm_cos_theta)
        + jnp.multiply(B, VkVm_sin_theta)
    ).T

    return P_mk


@jax.jit
def active_power_flow(
    V: jnp.DeviceArray, theta: jnp.DeviceArray, G: jnp.DeviceArray, B: jnp.DeviceArray
) -> jnp.DeviceArray:

    """
    Computes the power flow between all buses

    Args:
        - V -> Jax vector of N entries (N is the number of buses) containing the
        voltage magnitude of each bus
        - theta -> Jax vector of N entries (N is the number of buses) containing the
        voltage angle of each bus (in degrees)
        - G -> Jax matrix of NxN entries, containing the conductance of each transmission
        line
        - B -> Jax matrix of NxN entries, containing the susceptance of each transmission
        line

    Returns
        - P -> Jax matrix which the element ij is the active power flow from bus i to j
    """

    forward_flow = forward_active_power_flow(V, theta, G, B)
    backward_flow = backward_active_power_flow(V, theta, G, B)

    P = forward_flow + backward_flow

    return P


@jax.jit
def forward_reactive_power_flow(
    V: jnp.DeviceArray,
    theta: jnp.DeviceArray,
    G: jnp.DeviceArray,
    B: jnp.DeviceArray,
    Bsh: jnp.DeviceArray,
) -> jnp.DeviceArray:

    """
    Computes the power flow from buses which start indexes are lower than
    end indexes, i.e., the P_km flow.

    Args:
        - V -> Jax ROW vector of N entries (N is the number of buses) containing the
        voltage magnitude of each bus
        - theta -> Jax ROW vector of N entries (N is the number of buses) containing the
        voltage angle of each bus (in degrees)
        - G -> Jax matrix of NxN entries, containing the conductance of each transmission
        line
        - B -> Jax matrix of NxN entries, containing the susceptance of each transmission
        line
    Returns:
        - Q_km -> Jax matrix which the element ij is the reactive power flow from bus i
        to j

    The forward flow equation is

    Q_km = -(B+Bsh) V²_k + Bkm Vk Vm cos(theta_km) - Gkm Vk Vm sen(theta_km)

    Where k is the starting bus and m is the ending bus connected by a transmission line,
    theta_km = theta_k - theta_m

    An expression like theta_k - theta_m can be efficiently computed as:
    theta_km = theta - theta^T, by using Jax/Numpy broadcasting capabilities
    """

    # Auxiliary variables
    Vk = jnp.expand_dims(V, axis=0).T
    Vm = Vk.T

    theta_k = jnp.deg2rad(jnp.expand_dims(theta, axis=0)).T
    theta_m = theta_k.T

    # Operations
    theta_km = theta_k - theta_m
    VkVm = jnp.multiply(Vk, Vm)

    VkVm_cos_theta = jnp.multiply(VkVm, jnp.cos(theta_km))
    VkVm_sin_theta = jnp.multiply(VkVm, jnp.sin(theta_km))

    Q_km = (
        jnp.multiply(-(B + Bsh), jnp.square(Vk))
        + jnp.multiply(B, VkVm_cos_theta)
        - jnp.multiply(G, VkVm_sin_theta)
    )

    return Q_km


@jax.jit
def backward_reactive_power_flow(
    V: jnp.DeviceArray,
    theta: jnp.DeviceArray,
    G: jnp.DeviceArray,
    B: jnp.DeviceArray,
    Bsh: jnp.DeviceArray,
) -> jnp.DeviceArray:

    """
    Computes the power flow from buses which start indexes are lower than
    end indexes, i.e., the P_km flow.

    Args:
        - V -> Jax ROW vector of N entries (N is the number of buses) containing the
        voltage magnitude of each bus
        - theta -> Jax ROW vector of N entries (N is the number of buses) containing the
        voltage angle of each bus (in degrees)
        - G -> Jax matrix of NxN entries, containing the conductance of each transmission
        line
        - B -> Jax matrix of NxN entries, containing the susceptance of each transmission
        line
    Returns:
        - Q_mk -> Jax matrix which the element ij is the reactive power flow from bus i
        to j

    The forward flow equation is

    Q_mk = -(B+Bsh) V²_m + Bkm Vk Vm cos(theta_km) + Gkm Vk Vm sen(theta_km)

    Where k is the starting bus and m is the ending bus connected by a transmission line,
    theta_km = theta_k - theta_m

    An expression like theta_k - theta_m can be efficiently computed as:
    theta_km = theta - theta^T, by using Jax/Numpy broadcasting capabilities
    """

    # Auxiliary variables
    Vk = jnp.expand_dims(V, axis=0).T
    Vm = Vk.T

    theta_k = jnp.deg2rad(jnp.expand_dims(theta, axis=0)).T
    theta_m = theta_k.T

    # Operations
    theta_km = theta_k - theta_m
    VkVm = jnp.multiply(Vk, Vm)

    VkVm_cos_theta = jnp.multiply(VkVm, jnp.cos(theta_km))
    VkVm_sin_theta = jnp.multiply(VkVm, jnp.sin(theta_km))

    Q_mk = (
        jnp.multiply(-(B + Bsh), jnp.square(Vm))
        + jnp.multiply(B, VkVm_cos_theta)
        + jnp.multiply(G, VkVm_sin_theta)
    ).T

    return Q_mk


@jax.jit
def reactive_power_flow(
    V: jnp.DeviceArray,
    theta: jnp.DeviceArray,
    G: jnp.DeviceArray,
    B: jnp.DeviceArray,
    Bsh: jnp.DeviceArray,
) -> jnp.DeviceArray:

    """
    Computes the reactive power flow between all buses

    Args:
        - V -> Jax vector of N entries (N is the number of buses) containing the
        voltage magnitude of each bus
        - theta -> Jax vector of N entries (N is the number of buses) containing the
        voltage angle of each bus (in degrees)
        - G -> Jax matrix of NxN entries, containing the conductance of each transmission
        line
        - B -> Jax matrix of NxN entries, containing the susceptance of each transmission
        line

    Returns
        - Q -> Jax matrix which the element ij is the reactive power flow from bus i to j
    """

    forward_flow = forward_reactive_power_flow(V, theta, G, B, Bsh)
    backward_flow = backward_reactive_power_flow(V, theta, G, B, Bsh)

    Q = forward_flow + backward_flow

    return Q


@jax.jit
def active_power_losses(P: jnp.DeviceArray) -> jnp.DeviceArray:

    """
    This function calculates the active power losses in the transmission lines

    The loss on a line that connects buses i and j is given by:

    Ploss = Pij + Pji or Qloss = Qij + Qji

    args:
        - P: the Jax matrix containing all the forward and backward active power flows
    returns:
        - Ploss: the Jax matrix containing all the active power losses in the lines
    """

    Ploss = jnp.triu(P) + jnp.tril(P).T

    return Ploss


@jax.jit
def active_power_balance(
    P: jnp.DeviceArray, P_g: jnp.DeviceArray, P_c: jnp.DeviceArray
) -> jnp.DeviceArray:

    """
    This function calculates the active power balance for each bus in the system.
    The active power balance for the i-th bus is given by:

    DeltaP_i = P_G_i - P_C_i - sum P[i, :]
    """

    P_km = jnp.expand_dims(jnp.sum(P, axis=1), axis=0).T
    Delta_P = P_g - P_c - P_km

    return Delta_P


@jax.jit
def reactive_power_balance(
    Q: jnp.DeviceArray, Q_g: jnp.DeviceArray, Q_c: jnp.DeviceArray
) -> jnp.DeviceArray:

    Q_km = jnp.expand_dims(jnp.sum(Q, axis=1), axis=0).T
    Delta_Q = Q_g - Q_c - Q_km

    return Delta_Q


@jax.jit
def calculate_hydro_goal_deviation(
    P_g: jnp.DeviceArray, goal: jnp.DeviceArray
) -> float:

    """
    Calculates the total deviation between the generated power by hydro plants and their
    generation goal throughout the 24-hour time window

    args:
        - P_g: jax array of shape (N, 1, 24) that contains the power generated by the
        N hydro plants
        - goal: jax array of shape (N, 1) that contains the power generation goal
        of the N hydro plants
    returns:
        - deviation: float which is calculated by:

        deviation = sum(Pg_t) - goal for t in 25 periods and for each generator
    """

    # Sum the generations accross the 24 periods
    # Should have a shape of (N, 1)
    P_g_sum = jnp.sum(P_g, axis=-1)

    deviation = jnp.abs(P_g_sum - goal)
    deviation = jnp.sum(deviation).astype(float)

    return deviation


def objective_function(Pg_thermal: jnp.DeviceArray, extra_variables: dict) -> float:

    cost = generator_cost_function(
        Pg_thermal,
        a=extra_variables["a"],
        b=extra_variables["b"],
        c=extra_variables["c"],
    )

    return cost


def restrictions_function(Pg_thermal: jnp.DeviceArray, extra_variables: dict) -> float:

    return
