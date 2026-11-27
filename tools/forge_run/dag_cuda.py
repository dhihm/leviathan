"""Execute the captured Leviathan value DAG on NVIDIA CUDA."""

import argparse
import json
import sys

import torch

from dag_forge import (
    ATOL, MAX_SEED, RTOL, ForgeRunError, capture, forward_source, load_model, resolve_input,
)


def execute_captured_dag(graph, blob, layouts, x, device):
    """Execute exactly the immutable value DAG returned by capture()."""
    weights = torch.frombuffer(bytearray(blob), dtype=torch.float32).to(device)
    values = [x.to(device)]
    dense_layouts = {layout["nodeID"]: layout for layout in layouts}
    for node in graph["nodes"]:
        inputs = [values[index] for index in node["inputs"]]
        if node["op"] == "dense":
            layout = dense_layouts[node["id"]]
            weight = weights[layout["weightOffset"]:layout["biasOffset"]].view(
                layout["inDim"], layout["outDim"])
            bias = weights.narrow(0, layout["biasOffset"], layout["outDim"])
            value = torch.addmm(bias, inputs[0], weight)
        elif node["op"] == "relu":
            value = torch.relu(inputs[0])
        elif node["op"] == "add":
            value = torch.add(inputs[0], inputs[1])
        else:
            raise ForgeRunError(f"unexpected DAG operation {node['op']}")
        values.append(value)
    return values[graph["outputValue"]]


def run_cuda(model_path, example_input, user_input, seed=0):
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise ForgeRunError("CUDA backend requires NVIDIA hardware and a CUDA-enabled PyTorch")
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ForgeRunError(f"seed must be an int in 0..{MAX_SEED}")
    example = resolve_input(example_input, "example input")
    real = resolve_input(user_input, "input")
    model, input_dim, _, _ = load_model(
        model_path, seed, (("example input", example), ("input", real)))
    graph, blob, layouts, _ = capture(model, example, input_dim)
    try:
        device = torch.device("cuda:0")
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            with torch.cuda.device(device), torch.inference_mode():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                gpu = execute_captured_dag(graph, blob, layouts, real, device)
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
        finally:
            torch.set_float32_matmul_precision(precision)
        if not gpu.is_cuda or not 0 < elapsed < float("inf"):
            raise ForgeRunError("DAG execution produced no CUDA evidence")
        output = gpu.cpu()
        name = torch.cuda.get_device_name(device)
    except RuntimeError as exc:
        raise ForgeRunError(f"CUDA DAG execution failed: {exc}") from None
    reference = forward_source(model, real, "the input")
    if output.shape != reference.shape or not torch.allclose(output, reference, atol=ATOL, rtol=RTOL):
        raise ForgeRunError("CUDA DAG output disagrees with the source PyTorch module")
    return {"backend": "cuda (PyTorch)", "device": name, "operations": len(graph["nodes"]),
            "cudaEventMillis": elapsed, "output": output.tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="TRUSTED local Python model file (executed)")
    parser.add_argument("--example-input", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = run_cuda(args.model, args.example_input, args.input, args.seed)
    except ForgeRunError as exc:
        print(f"dag_cuda: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
