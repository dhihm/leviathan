"""Real Metal cache acceptance coverage for the declarative BUILD CLI."""
import hashlib
import json
import subprocess
import struct
import sys

from tools.forge_run.test_dag_runtime import HELPER, TOOL_DIR, residual, work


def test_declared_build_cache_validation(residual, tmp_path):
    _, _, baseline, _, _ = residual
    graph = baseline / "model.graph.json"
    weights = baseline / "model.weights.f32"
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    declaration = dict(format="leviathan-declared-dag-1", inputShape=[1, 32], outputShape=[1, 32],
                       graphSha256=sha(graph), weightsSha256=sha(weights))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(declaration))
    example, changed = (tmp_path / name for name in ("example.json", "changed.json"))
    example.write_text(json.dumps([[0.25] * 32]))
    changed.write_text(json.dumps([[0.75] * 32]))
    cache = tmp_path / "private-cache"
    base = [sys.executable, str(TOOL_DIR / "dag_build_declared.py"), "--helper", str(HELPER),
            "--graph", str(graph), "--weights", str(weights), "--contract", str(contract),
            "--cache", str(cache)]

    def build(name, actual=example, calibration=example, flags=(), success=True):
        out = tmp_path / name
        result = subprocess.run(base + ["--out", str(out), "--example-input", str(calibration),
                                        "--input", str(actual), *flags],
                                capture_output=True, text=True, timeout=60)
        if not success:
            assert result.returncode == 2 and not out.exists()
            return result.stderr
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    cold, warm = build("cold"), build("warm")
    assert not cold["cache"]["hit"] and warm["cache"]["hit"]
    assert cold["run"]["output"] == warm["run"]["output"]
    changed_run = build("changed", actual=changed)
    assert changed_run["cache"]["hit"] and changed_run["run"]["output"] != warm["run"]["output"]
    assert not build("new-calibration", calibration=changed)["cache"]["hit"]
    assert not build("new-option", flags=("--diagnostic-fail-optimizations",))["cache"]["hit"]

    record = cache / (cold["cache"]["key"] + ".json")
    record.write_text('{"key": "corrupt"}')
    recovered = build("recovered")
    assert not recovered["cache"]["hit"] and recovered["cache"]["reason"].startswith("rejected:")
    assert recovered["run"]["output"] == cold["run"]["output"]
    assert build("repaired-hit")["cache"]["hit"]

    cached = json.loads(record.read_text())
    metadata = json.loads(cached["bundle"])
    metadata["hardware"]["deviceName"] = "other GPU"
    cached["bundle"] = json.dumps(metadata)
    cached["sha256"] = hashlib.sha256(cached["bundle"].encode()).hexdigest()
    record.write_text(json.dumps(cached))
    rebound = build("rebound")
    assert not rebound["cache"]["hit"] and "hardware mismatch" in rebound["cache"]["reason"]
    assert rebound["run"]["output"] == cold["run"]["output"]

    altered = tmp_path / "altered.f32"
    data = weights.read_bytes()
    changed_weight = struct.unpack_from("<f", data)[0] + 1
    altered.write_bytes(struct.pack("<f", changed_weight) + data[4:])
    base[base.index("--weights") + 1] = str(altered)
    assert "content hash mismatch" in build("unbound-weights", success=False)
    contract.write_text(json.dumps(dict(declaration, weightsSha256=sha(altered))))
    updated = build("updated-weights")
    assert not updated["cache"]["hit"] and updated["run"]["output"] != cold["run"]["output"]

    contract.write_text(json.dumps(dict(declaration, graphSha256="0" * 64)))
    assert "content hash mismatch" in build("invalid", success=False)
