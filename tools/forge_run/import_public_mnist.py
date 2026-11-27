"""Import the pinned MIT-licensed Badak MNIST checkpoint and reference rows."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load

from dag_build_declared import FORMAT, digest, encode, read

REVISION = "ff9e117d7f1f3902e55c68b29eda74700c9282c5"
CHECKPOINT_SHA = "6049a34ee22aa69a72c7e6b7d5c69a044df6192ae1d154bca6a1b0bc2f8a5b44"
REFERENCE_SHA = "872f346f792f34bbf4fc8a274adbe9823c102a8af45adc390dce3a7fe3c9d0b3"


def import_checkpoint(checkpoint, reference, out):
    model_data, reference_data = read(checkpoint), read(reference)
    if digest(model_data) != CHECKPOINT_SHA or digest(reference_data) != REFERENCE_SHA:
        raise ValueError("checkpoint or reference SHA-256 does not match the pinned release")
    model, known = load(model_data), load(reference_data)
    graph = encode({"format": "leviathan-value-dag-1", "inputDim": 784, "outputValue": 3,
                    "nodes": [{"id": 1, "op": "dense", "inputs": [0], "outputDim": 128},
                              {"id": 2, "op": "relu", "inputs": [1], "outputDim": 128},
                              {"id": 3, "op": "dense", "inputs": [2], "outputDim": 10}]})
    packed = b"".join(np.asarray(array, dtype="<f4").tobytes()
                      for array in (model["fc1.weight"].T, model["fc1.bias"],
                                    model["fc2.weight"].T, model["fc2.bias"]))
    contract = encode({"format": FORMAT, "inputShape": [1, 784], "outputShape": [1, 10],
                       "graphSha256": digest(graph), "weightsSha256": digest(packed)})
    provenance = encode({"model": "building-an-inference-engine-from-scratch/mnist-mlp",
                         "revision": REVISION, "license": "MIT", "checkpointSha256": CHECKPOINT_SHA,
                         "referenceSha256": REFERENCE_SHA})
    out.mkdir()
    for name, data in (("graph.json", graph), ("weights.f32", packed), ("contract.json", contract),
                       ("rows.json", encode(known["input"].tolist())),
                       ("logits.json", encode(known["logits"].tolist())), ("provenance.json", provenance)):
        (out / name).write_bytes(data)
    for i, row in enumerate(known["input"]):
        (out / f"input-{i}.json").write_text(json.dumps([row.tolist()]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("checkpoint", "reference", "out"):
        parser.add_argument("--" + option, required=True, type=Path)
    args = parser.parse_args()
    import_checkpoint(args.checkpoint, args.reference, args.out)
