#!/usr/bin/env python3
"""Positive/negative tests for the strict torch.export frontend.

Runs tools/torch_frontend/export_torch_graph.py as a subprocess on the
sample models (stdlib asserts, no test framework dependency):

- positives must exit 0; the manifest reference output is re-verified
  INDEPENDENTLY here by rebuilding the model and evaluating it on the same
  rank-2 contract input, and exported weight bytes are spot-checked against
  the model's own parameters;
- negatives must exit nonzero with an actionable message naming the reason
  (op name, structure, dtype, hooks, shared parameters, semantics
  divergence) and must leave NO output files behind.

Usage: python3 test_export.py [path-to-export_torch_graph.py]
Exit code 0 iff every check passes.
"""

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXPORTER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "export_torch_graph.py")
EXAMPLES = os.path.join(REPO, "examples", "torch_export")

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def run_exporter(model_path, prefix, seed="0"):
    proc = subprocess.run(
        [sys.executable, EXPORTER, model_path, prefix, seed],
        capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def output_files(prefix):
    return [p for p in (prefix + ".graph.json", prefix + ".weights.f32",
                        prefix + ".manifest.json") if os.path.exists(p)]


def load_module(path):
    spec = importlib.util.spec_from_file_location("sample", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def positive_case(tmp, name, model_file, verify):
    prefix = os.path.join(tmp, name)
    model_path = os.path.join(EXAMPLES, model_file)
    code, log = run_exporter(model_path, prefix)
    check(f"{name}: export exits 0", code == 0, log[-500:])
    if code != 0:
        return None
    manifest = json.load(open(prefix + ".manifest.json"))
    graph = json.load(open(prefix + ".graph.json"))
    blob = open(prefix + ".weights.f32", "rb").read()

    check(f"{name}: manifest format is weights-1",
          manifest["format"] == "leviathan-forge-weights-1")
    prov = manifest.get("torchExport")
    check(f"{name}: torchExport block is labeled advisory-only",
          bool(prov) and prov.get("advisoryOnly") is True)
    check(f"{name}: provenance records strict export",
          bool(prov) and prov["strict"] is True)
    check(f"{name}: advisory hashes match emitted bytes",
          prov["graphSha256"] == hashlib.sha256(
              open(prefix + ".graph.json", "rb").read()).hexdigest()
          and prov["weightsSha256"] == hashlib.sha256(blob).hexdigest())
    check(f"{name}: aten ops are exactly linear/relu",
          set(prov["atenOps"]) <= {"aten.linear.default", "aten.relu.default"}
          and "aten.linear.default" in prov["atenOps"])
    check(f"{name}: weights content hash matches manifest",
          manifest["weightsSha256"] == hashlib.sha256(blob).hexdigest())
    check(f"{name}: float count matches blob",
          manifest["floatCount"] == len(blob) // 4)

    # INDEPENDENT parity: rebuild the model here and evaluate it directly on
    # the rank-2 contract input.
    sample = load_module(model_path)
    model = sample.build(0).eval()
    ref_in = torch.tensor(manifest["refInput"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        want = model(ref_in).flatten().tolist()
    worst = max(abs(a - b) for a, b in zip(want, manifest["refOutput"]))
    check(f"{name}: manifest reference equals independent module output",
          worst == 0.0, f"worst abs diff {worst}")
    verify(sample, model, manifest, graph, blob)
    return manifest


with tempfile.TemporaryDirectory(prefix="leviathan-pt-frontend-") as tmp:
    def verify_custom(sample, model, manifest, graph, blob):
        # The graph is captured from forward semantics, not child labels:
        # the module has 3 child Linear modules; forward calls stem, both
        # Block.projection linears, and head via F.linear — 4 dense ops.
        dense = sum(1 for op in graph["ops"] if isinstance(op, dict))
        linears = sum(1 for m in model.modules() if type(m) is torch.nn.Linear)
        check("custom-forward: graph dense ops differ from child labels",
              dense == 4 and linears == 5,
              f"dense {dense}, child linears {linears}")
        # Exported weight bytes for the FIRST executed linear equal the
        # model's stem.weight transposed to [in, out].
        first = manifest["ops"][0]
        floats = struct.unpack(f"<{len(blob) // 4}f", blob)
        w = torch.tensor(floats[first["weightFloatOffset"]:
                                first["weightFloatOffset"] + first["weightFloatCount"]]
                         ).reshape(first["inDim"], first["outDim"])
        stem_w = dict(model.named_parameters())["stem.weight"]
        check("custom-forward: first exported weight is stem.weight^T",
              torch.equal(w, stem_w.t().contiguous()))
        check("custom-forward: graph follows the model",
              graph["inputDim"] == sample.INPUT_DIM
              and graph["ops"][-1] == {"dense": {"outDim": 8}})

    m1 = positive_case(tmp, "custom", "custom_forward_mlp.py", verify_custom)

    def verify_end_relu(sample, model, manifest, graph, blob):
        check("end-relu: graph terminates with a relu op",
              graph["ops"][-1] == "relu"
              and graph["ops"][-2] == {"dense": {"outDim": 12}})

    m2 = positive_case(tmp, "endrelu", "end_relu_mlp.py", verify_end_relu)

    # Determinism: same seed twice -> identical weight digests.
    if m1:
        p2 = os.path.join(tmp, "custom2")
        c2, _ = run_exporter(os.path.join(EXAMPLES, "custom_forward_mlp.py"), p2, "0")
        if c2 == 0:
            check("same seed is deterministic",
                  m1["weightsSha256"] == json.load(
                      open(p2 + ".manifest.json"))["weightsSha256"])
        else:
            check("same seed is deterministic", False, "repeat export failed")

    # ---- negatives: each must fail closed, name the reason, and leave
    # no output files behind ----
    def expect_reject(name, model_file, needle):
        path = os.path.join(EXAMPLES, "negatives", model_file)
        p = os.path.join(tmp, "neg-" + name.replace(" ", "-"))
        c, log = run_exporter(path, p)
        check(f"negative {name} rejected", c != 0, f"exit 0: {log[-300:]}")
        check(f"negative {name} message is actionable",
              needle in log, f"missing '{needle}' in: {log[-300:]}")
        check(f"negative {name} wrote no files", not output_files(p),
              f"files: {output_files(p)}")

    expect_reject("sin", "neg_sin.py", "aten.sin")
    expect_reject("residual add", "neg_residual.py", "aten.add")
    expect_reject("multiple outputs", "neg_multi_output.py", "exactly one output")
    expect_reject("float16 dtype", "neg_dtype.py", "float32")
    expect_reject("hook", "neg_hook.py", "hook")
    expect_reject("reshape", "neg_reshape.py", "aten.view")
    expect_reject("shared parameter", "neg_shared.py", "shared")
    # Data-dependent branch: torch.export itself fails (fail-closed) or the
    # bounded differential semantics gate catches the specialized branch.
    p = os.path.join(tmp, "neg-branch")
    c, log = run_exporter(os.path.join(EXAMPLES, "negatives", "neg_branch.py"), p)
    check("negative data-dependent branch rejected", c != 0, f"exit 0: {log[-300:]}")
    check("branch rejection names semantics/export failure",
          c != 0 and ("disagrees with the original module" in log
                      or "torch.export failed" in log), log[-300:])
    check("negative branch wrote no files", not output_files(p))

print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
