"""Execute the captured Leviathan value DAG on NVIDIA CUDA."""

import argparse
import json
import sys
import time

import torch

from dag_forge import (
    ATOL, MAX_SEED, RTOL, ForgeRunError, capture, forward_source, load_model, resolve_input,
)


def execute_captured_dag(graph, blob, layouts, x, device):
    """Execute exactly the immutable value DAG returned by capture()."""
    weights, dense_layouts = prepare_captured_dag(blob, layouts, device)
    return execute_prepared_dag(graph, weights, dense_layouts, x)


def prepare_captured_dag(blob, layouts, device):
    weights = torch.frombuffer(bytearray(blob), dtype=torch.float32).to(device)
    dense_layouts = {layout["nodeID"]: layout for layout in layouts}
    return weights, dense_layouts


def execute_prepared_dag(graph, weights, dense_layouts, x):
    values = [x.to(weights.device)]
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
    return run_cuda_many(model_path, example_input, [user_input], seed)["results"][0]


def run_cuda_many(model_path, example_input, user_inputs, seed=0):
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise ForgeRunError("CUDA backend requires NVIDIA hardware and a CUDA-enabled PyTorch")
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ForgeRunError(f"seed must be an int in 0..{MAX_SEED}")
    if not user_inputs:
        raise ForgeRunError("at least one input is required")
    started = time.perf_counter()
    example = resolve_input(example_input, "example input")
    inputs = [resolve_input(value, "input") for value in user_inputs]
    model, input_dim, _, _ = load_model(
        model_path, seed, (("example input", example), *(("input", x) for x in inputs)))
    graph, blob, layouts, _ = capture(model, example, input_dim)
    captured = time.perf_counter()
    device = torch.device("cuda:0")
    precision = torch.get_float32_matmul_precision()
    results = []
    try:
        with torch.cuda.device(device):
            weights, dense_layouts = prepare_captured_dag(blob, layouts, device)
            torch.cuda.synchronize(device)
            prepared = time.perf_counter()
            name = torch.cuda.get_device_name(device)
            for real in inputs:
                request_start = time.perf_counter()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                try:
                    torch.set_float32_matmul_precision("highest")
                    start.record()
                    gpu = execute_prepared_dag(graph, weights, dense_layouts, real)
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end)
                finally:
                    torch.set_float32_matmul_precision(precision)
                if not gpu.is_cuda or not 0 < elapsed < float("inf"):
                    raise ForgeRunError("DAG execution produced no CUDA evidence")
                output = gpu.cpu()
                reference = forward_source(model, real, "the input")
                if output.shape != reference.shape or not torch.allclose(output, reference, atol=ATOL, rtol=RTOL):
                    raise ForgeRunError("CUDA DAG output disagrees with the source PyTorch module")
                results.append({"backend": "cuda (PyTorch)", "device": name,
                                "operations": len(graph["nodes"]), "cudaEventMillis": elapsed,
                                "requestWallMillis": (time.perf_counter() - request_start) * 1000,
                                "output": output.tolist()})
    except RuntimeError as exc:
        raise ForgeRunError(f"CUDA DAG execution failed: {exc}") from None
    return {"captureMillis": (captured - started) * 1000,
            "devicePreparationMillis": (prepared - captured) * 1000,
            "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="TRUSTED local Python model file (executed)")
    parser.add_argument("--example-input", required=True)
    parser.add_argument("--input", required=True, action="append", help="repeat to reuse device weights")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = run_cuda_many(args.model, args.example_input, args.input, args.seed)
        if len(args.input) == 1:
            report = report["results"][0]
    except ForgeRunError as exc:
        print(f"dag_cuda: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
