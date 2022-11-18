from typing import Callable, List, Union
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
from cyipopt import minimize_ipopt
import pdb


def optimize(
    x0: jnp.DeviceArray,
    objective: Callable,
    constraints: List[dict],
    bounds: List,
    extra_variables: Union[dict, None] = None,
):

    """
    Executes the Interior Point Method with logarithmic barrier to solve a constrained non-linear
    optimization problem.

    Args:
        - x0: initial point of the problem's variables
        - objective: objective function
        - constraints: list of dictionaries with the following format:
            - {
                "type": "eq" or "ineq" -> selects the type of the restriction,
                "fun": callable Jax function,
                "jac": callable jacobian of the restriction function,
                "hess": callable Hessian vector product
            }
        - extra_variables: these should contain all the extra variables needed for the
        IPOPT method to work on the Optimal Power Flow problem:
            - {
                "a": JAX vector containing the quadratic cost parameter of the generators
                ...
            }
    """

    J = jax.jit(jax.jacfwd(objective, argnums=0))

    @jax.jit
    def H(x, v, extra_variables=None):
        return jax.hessian(objective, argnums=0)(x, extra_variables) * v

    res = minimize_ipopt(
        fun=objective,
        jac=J,
        hess=H,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        options={"disp": 5},
    )
