"""Example LeviathanForge model definition (Python frontend contract).

A tiny dense/relu MLP architecture. Weights are synthetic (seeded) in
milestone 0 — see forge_demo.py for the honest scope.
"""

MODEL = {
    "inputDim": 64,
    "ops": [
        {"dense": {"outDim": 128}},
        "relu",
        {"dense": {"outDim": 64}},
        "relu",
        {"dense": {"outDim": 8}},
    ],
}
