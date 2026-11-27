"""Execute the captured Leviathan value DAG on NVIDIA CUDA."""

import argparse
import json
import sys
from pathlib import Path
import time

import torch

import dag_build_declared as declared

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
    require_cuda()
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
    return execute_cuda_requests(graph, blob, layouts, inputs,
                                 lambda real: forward_source(model, real, "the input"), started)


def require_cuda():
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise ForgeRunError("CUDA backend requires NVIDIA hardware and a CUDA-enabled PyTorch")


def run_declared_cuda(contract, graph_path, weights_path, user_inputs):
    started = time.perf_counter()
    specification, graph, _, blob, weights, layouts = declared.load_model(
        Path(contract), Path(graph_path), Path(weights_path))
    inputs = [torch.from_numpy(declared.read_input(Path(path), graph["inputDim"])).reshape(1, -1)
              for path in user_inputs]

    softmax = specification.get("outputActivation") == "softmax"

    def reference(real):
        result = declared.reference(graph, weights, layouts, real.numpy()[0])
        if softmax:
            result = declared.softmax_reference(result)
        return torch.from_numpy(result).reshape(1, -1)

    require_cuda()
    return execute_cuda_requests(graph, blob, layouts, inputs, reference, started, softmax=softmax)


def execute_cuda_requests(graph, blob, layouts, inputs, source_reference, started, softmax=False):
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
                    if softmax:
                        gpu = torch.softmax(gpu, dim=-1)
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end)
                finally:
                    torch.set_float32_matmul_precision(precision)
                if not gpu.is_cuda or not 0 < elapsed < float("inf"):
                    raise ForgeRunError("DAG execution produced no CUDA evidence")
                output = gpu.cpu()
                reference = source_reference(real)
                if output.shape != reference.shape or not torch.allclose(output, reference, atol=ATOL, rtol=RTOL):
                    raise ForgeRunError("CUDA DAG output disagrees with the independent CPU reference")
                results.append({"backend": "cuda (PyTorch)", "device": name,
                                "operations": len(graph["nodes"]) + int(softmax), "cudaEventMillis": elapsed,
                                "requestWallMillis": (time.perf_counter() - request_start) * 1000,
                                "output": output.tolist()})
    except RuntimeError as exc:
        raise ForgeRunError(f"CUDA DAG execution failed: {exc}") from None
    return {"captureMillis": (captured - started) * 1000,
            "devicePreparationMillis": (prepared - captured) * 1000,
            "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", help="TRUSTED local Python model file (executed)")
    mode.add_argument("--contract", help="versioned declarative model JSON")
    parser.add_argument("--graph")
    parser.add_argument("--weights")
    parser.add_argument("--example-input")
    parser.add_argument("--input", required=True, action="append", help="repeat to reuse device weights")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    try:
        if args.contract is not None:
            if not args.graph or not args.weights or args.example_input is not None or args.seed is not None:
                parser.error("--contract requires --graph and --weights and accepts no export options")
            report = run_declared_cuda(args.contract, args.graph, args.weights, args.input)
        else:
            if not args.example_input or args.graph is not None or args.weights is not None:
                parser.error("--model requires --example-input")
            report = run_cuda_many(args.model, args.example_input, args.input,
                                   args.seed if args.seed is not None else 0)
        if len(args.input) == 1:
            report = report["results"][0]
    except (ForgeRunError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"dag_cuda: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
