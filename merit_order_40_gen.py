import jax.numpy as jnp
import pandas as pd
import numpy as np
import pdb

from power_flow.merit_order_dispatch import merit_order_dispatch_algorithm

Pg_min = jnp.array(
    [
        36,
        36,
        60,
        80,
        47,
        68,
        110,
        135,
        135,
        130,
        94,
        94,
        125,
        125,
        125,
        125,
        220,
        220,
        242,
        242,
        254,
        254,
        254,
        254,
        254,
        254,
        10,
        10,
        10,
        47,
        60,
        60,
        60,
        90,
        90,
        90,
        25,
        25,
        25,
        242,
    ],
    dtype=jnp.float32,
)

Pg_max = jnp.array(
    [
        114,
        114,
        120,
        190,
        97,
        140,
        300,
        300,
        300,
        300,
        375,
        375,
        500,
        500,
        500,
        500,
        500,
        500,
        550,
        550,
        550,
        550,
        550,
        550,
        550,
        550,
        150,
        150,
        150,
        97,
        190,
        190,
        190,
        200,
        200,
        200,
        110,
        110,
        110,
        550,
    ],
    dtype=jnp.float32,
)

a = jnp.array(
    [
        0.00690,
        0.00690,
        0.02028,
        0.00942,
        0.01140,
        0.01142,
        0.00357,
        0.00492,
        0.00573,
        0.00605,
        0.00515,
        0.00569,
        0.00421,
        0.00752,
        0.00752,
        0.00752,
        0.00313,
        0.00313,
        0.00313,
        0.00313,
        0.00298,
        0.00298,
        0.00284,
        0.00284,
        0.00277,
        0.00277,
        0.52124,
        0.52124,
        0.52124,
        0.01140,
        0.00160,
        0.00160,
        0.00160,
        0.00010,
        0.00010,
        0.00010,
        0.01610,
        0.01610,
        0.01610,
        0.00313,
    ]
)

b = jnp.array(
    [
        6.73,
        6.73,
        7.07,
        8.18,
        5.35,
        8.05,
        8.03,
        6.99,
        6.60,
        12.9,
        12.9,
        12.8,
        12.5,
        8.84,
        8.84,
        8.84,
        7.97,
        7.95,
        7.97,
        7.97,
        6.63,
        6.63,
        6.66,
        6.66,
        7.10,
        7.10,
        3.33,
        3.33,
        3.33,
        5.35,
        6.43,
        6.43,
        6.43,
        8.95,
        8.62,
        8.62,
        5.88,
        5.88,
        5.88,
        7.97,
    ]
)

demand = 10500.0


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