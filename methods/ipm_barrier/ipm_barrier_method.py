from dataclasses import dataclass
from typing import Callable, Union
import jax
import jax.numpy as jnp
from jax.numpy.linalg import inv
import pdb
from functools import partial


# @partial(jax.jit, static_argnums=(1, 2))
def log_barrier(
    x: jnp.DeviceArray,
    restrictions: Callable,
    extra_variables: Union[dict, None] = None,
) -> float:

    """
    This function calculates the logarithmic barrier for the
    Interior Point Method, which can be expressed as:

    P(x) = - sum( log(-h_i(x)) )

    In which h_i represents the i-th restriction of the problem and x
    represents its input.

    args:
        - x: a JAX array containing the input to the restrictions
        - restrictions: a Python function which evaluates the restrictions
        given an input x
    returns:
        - barrier: a float value calculated using the expression above
    """
    res = restrictions(x, extra_variables)

    barrier = jnp.log(-res)
    barrier = -jnp.sum(barrier)
    barrier = barrier.astype(float)

    return barrier


def _newton_method_internal_op(
    xN: jnp.DeviceArray,
    J: Callable,
    H: Callable,
    t: float = 0.1,
    extra_variables: Union[dict, None] = None,
):

    Hinv = inv(H(xN, t, extra_variables))
    j = J(xN, t, extra_variables)

    xN1 = xN - jnp.dot(Hinv, j)
    pdb.set_trace()

    mismatch = jnp.linalg.norm(xN1 - xN)

    return xN1, mismatch


def newton_method(
    x0: jnp.DeviceArray,
    f: Callable,
    J: Callable,
    H: Callable,
    t: float = 0.1,
    extra_variables: Union[dict, None] = None,
    it_max: int = 100,
    eps: float = 1e-4,
) -> jnp.DeviceArray:

    xN = None
    it = 0
    mismatch = jnp.inf

    while mismatch > eps:

        print(f"Newton Method - t = {it} \t error = {mismatch}")

        if it == it_max:
            break

        if xN is None:
            xN = x0.copy()

        xN, mismatch = _newton_method_internal_op(xN, J, H, t, extra_variables)

        it += 1

    return xN


def optimize(
    x0: jnp.DeviceArray,
    objective: Callable,
    restrictions: Callable,
    num_restrictions: int,
    t: float,
    eps: float = 1e-5,
    it_max: int = 1000,
    v: float = 0.01,
    extra_variables: Union[dict, None] = None,
):

    """
    Executes the Interior Point Method with logarithmic barrier to solve a constrained non-linear
    optimization problem.

    Args:
        - x0: initial point of the problem's variables
        - objective: objective function
        - restrictions: function that executes all the restrictions and return an array with their
        values
        - num_restrictions: number of restrictions, the 'm' parameter in the method
        - t: initial value for the 't' parameter in the method
        - eps: tolerance for the duality gap in the method
        - it_max: maximum number of iterations for the Newton method step
        - v: parameter of the barrier function
        - extra_variables: these should contain all the extra variables needed for the
        IPM method to work on the Optimal Power Flow problem:
            - {
                "a": JAX vector containing the quadratic cost parameter of the generators
                ...
            }

    """

    # Create a function that calls both the objective and the restrictions
    def F(x: jnp.DeviceArray, t: float, extra_variables: dict = None) -> float:
        fun = objective(x, extra_variables) + t * log_barrier(
            x, restrictions, extra_variables
        )
        fun = fun.sum().astype(float)
        return fun

    J = jax.jacfwd(F, argnums=0)
    H = jax.hessian(F, argnums=0)

    x = x0.copy()
    x_list = [x]
    f_list = [objective(x, extra_variables)]
    res_list = [restrictions(x, extra_variables)]
    duality_gap = num_restrictions / t
    duality_gap_list = [duality_gap]

    # Initial iteration
    it = 0

    while duality_gap > eps:
        print(
            f"Iteration: {it} \t Duality Gap: {duality_gap} \t f(x)={f_list[-1]} \t res(x)={res_list[-1]}"
        )

        # Newton method
        x = newton_method(x, F, J, H, t, it_max, eps=1e-4)

        t = t + (t / (13 * jnp.sqrt(v)))
        duality_gap = num_restrictions / t

        pdb.set_trace()

        duality_gap_list.append(duality_gap)

        x_list.append(x)
        f_list.append(objective(x))
        res_list.append(restrictions(x))
        it = it + 1

    pdb.set_trace()

    return x, x_list, f_list, res_list, duality_gap_list
