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