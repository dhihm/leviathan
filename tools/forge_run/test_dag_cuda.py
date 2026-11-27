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

    def test_cuda_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            example = Path(directory) / "example.json"
            actual = Path(directory) / "actual.json"
            example.write_text(json.dumps([[0.25] * 32]))
            actual.write_text(json.dumps([[0.75] * 32]))
            result = subprocess.run(
                [sys.executable, str(TOOL / "dag_cuda.py"), "--model", str(MODEL),
                 "--example-input", str(example), "--input", str(actual)],
                capture_output=True, text=True, timeout=90,
            )
        if not torch.cuda.is_available() or torch.version.cuda is None:
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("CUDA-enabled PyTorch", result.stderr)
            self.assertFalse(result.stdout)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["backend"], "cuda (PyTorch)")
            self.assertGreater(report["cudaEventMillis"], 0)
            self.assertGreater(report["operations"], 0)
            self.assertEqual(len(report["output"][0]), 32)


if __name__ == "__main__":
    unittest.main()
