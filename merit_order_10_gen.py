import jax.numpy as jnp
import pandas as pd
import numpy as np

from power_flow.merit_order_dispatch import merit_order_dispatch_algorithm

Pg_min = jnp.array(
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    dtype=jnp.float32,
)

Pg_max = jnp.array(
    [625, 625, 625, 625, 625, 625, 625, 625, 625, 625],
    dtype=jnp.float32,
)

a = jnp.array(
    [0.008, 0.0096, 0.0098, 0.0102, 0.0088, 0.009, 0.0099, 0.0102, 0.0087, 0.0098]
)

b = jnp.array([8, 6.4, 6.3, 7, 6.5, 6.4, 6.2, 7.8, 6.5, 6.4])

demand = 4000.0


dispatch, lambd = merit_order_dispatch_algorithm(
    Pg_max=Pg_max, Pg_min=Pg_min, a=a, b=b, demand=demand
)

print(f"Marginal cost = {lambd}")

results = []
for i in range(len(dispatch)):
    results.append({"generator": i, "dispatch": dispatch[i]})

results_table = pd.DataFrame(results)
print(results_table)

np.testing.assert_approx_equal(jnp.sum(dispatch).item(), demand, significant=1)
