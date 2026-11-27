import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

TOOL = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL))
import dag_cuda
import dag_forge
MODEL = TOOL / "examples" / "residual_block.py"


class DAGCUDATest(unittest.TestCase):
    def test_captured_residual_executes_from_ir(self):
        x = torch.linspace(-1, 1, 32).reshape(1, -1)
        model, dim, _, _ = dag_forge.load_model(MODEL, 0, (("input", x),))
        graph, blob, layouts, _ = dag_forge.capture(model, x, dim)
        result = dag_cuda.execute_captured_dag(graph, blob, layouts, x, torch.device("cpu"))
        torch.testing.assert_close(result, model(x), atol=1e-4, rtol=1e-4)

    def test_prepared_tensors_reused_for_changed_input(self):
        x = torch.full((1, 32), 0.25)
        model, dim, _, _ = dag_forge.load_model(MODEL, 0, (("input", x),))
        graph, blob, layouts, _ = dag_forge.capture(model, x, dim)
        weights, bound_layouts = dag_cuda.prepare_captured_dag(blob, layouts, torch.device("cpu"))
        original = weights.clone()
        outputs = [dag_cuda.execute_prepared_dag(graph, weights, bound_layouts, value)
                   for value in (x, x + 0.5, x)]
        self.assertTrue(torch.equal(outputs[0], outputs[2]))
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        self.assertTrue(torch.equal(weights, original))
        torch.testing.assert_close(outputs[1], model(x + 0.5), atol=1e-4, rtol=1e-4)

    def test_cuda_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            example = Path(directory) / "example.json"
            actual = Path(directory) / "actual.json"
            example.write_text(json.dumps([[0.25] * 32]))
            actual.write_text(json.dumps([[0.75] * 32]))
            command = [sys.executable, str(TOOL / "dag_cuda.py"), "--model", str(MODEL),
                       "--example-input", str(example), "--input", str(actual)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            batch = subprocess.run(command + ["--input", str(example), "--input", str(actual)],
                                   capture_output=True, text=True, timeout=90)
        if not torch.cuda.is_available() or torch.version.cuda is None:
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("CUDA-enabled PyTorch", result.stderr)
            self.assertFalse(result.stdout)
            self.assertEqual(batch.returncode, 2, batch.stderr)
            self.assertFalse(batch.stdout)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["backend"], "cuda (PyTorch)")
            self.assertGreater(report["cudaEventMillis"], 0)
            self.assertGreater(report["operations"], 0)
            self.assertEqual(len(report["output"][0]), 32)
            self.assertEqual(batch.returncode, 0, batch.stderr)
            prepared = json.loads(batch.stdout)
            self.assertGreater(prepared["devicePreparationMillis"], 0)
            self.assertGreater(prepared["captureMillis"], 0)
            first, changed, repeated = prepared["results"]
            self.assertEqual(first["output"], report["output"])
            self.assertEqual(repeated["output"], report["output"])
            self.assertNotEqual(changed["output"], report["output"])
            self.assertTrue(all(row["cudaEventMillis"] > 0 for row in prepared["results"]))


if __name__ == "__main__":
    unittest.main()
