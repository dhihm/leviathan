"""Import the full pinned Apache-2.0 dacorvo MNIST probability model."""

import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import load

from dag_build_declared import SOFTMAX_FORMAT, digest, encode, read

REVISION = "7ed1e96d182395fd7380ea8495b9eb2364ce9dc1"
CHECKPOINT_SHA = "59dfd2968fc0c60192c8e41ecff29f9271ec625684c9c3368ceae4166d089aec"


def import_checkpoint(path, out):
    data = read(path, 2 << 20)
    if digest(data) != CHECKPOINT_SHA:
        raise ValueError("checkpoint SHA-256 does not match the pinned release")
    model = load(data)
    nodes, chunks = [], []
    for name, width in (("input_layer", 256), ("mid_layer", 256), ("output_layer", 10)):
        nodes.append({"id": len(nodes) + 1, "op": "dense", "inputs": [len(nodes)], "outputDim": width})
        chunks.extend(np.asarray(value, dtype="<f4").tobytes()
                      for value in (model[name + ".weight"].T, model[name + ".bias"]))
        if name != "output_layer":
            nodes.append({"id": len(nodes) + 1, "op": "relu", "inputs": [len(nodes)], "outputDim": width})
    graph = encode({"format": "leviathan-value-dag-1", "inputDim": 784, "outputValue": len(nodes),
                    "nodes": nodes})
    weights = b"".join(chunks)
    contract = encode({"format": SOFTMAX_FORMAT, "inputShape": [1, 784], "outputShape": [1, 10],
                       "outputActivation": "softmax", "graphSha256": digest(graph),
                       "weightsSha256": digest(weights)})
    provenance = encode({"model": "dacorvo/mnist-mlp", "revision": REVISION,
                         "license": "Apache-2.0", "checkpointSha256": CHECKPOINT_SHA})
    out.mkdir()
    for name, data in (("graph.json", graph), ("weights.f32", weights),
                       ("contract.json", contract), ("provenance.json", provenance)):
        (out / name).write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    import_checkpoint(args.checkpoint, args.out)
