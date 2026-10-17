"""Attention-output projection as a LeviathanForge Q4_0 operator input.

This file is an OPERATOR-DSL input (Q4MatvecOp): it describes ONE bounded
quantized operator — the per-layer attention-output matrix-vector product
y[row] = dot(W_row, x) with W in row-major Q4_0 blocks — for the automatic
Q4 lowering pipeline (op JSON -> Swift validation -> shape+hardware-derived
schedule candidates -> emitted Metal -> FP64 oracle gate). It is NOT a
torch.export of a whole model and not a general Python compiler claim.

The "attention-output-family" view states that the bound tensor is one
layer's strict blk.<digits>.attn_output.weight 2-D Q4_0 tensor and that the
lowering may serve the sibling layers of the same family that share the
EXACT bound (in_dim, out_dim); every other shape falls through to the
incumbent kernels.
"""

from q4_op_model import Q4MatvecOp


IN_DIM = 4096
OUT_DIM = 2816
TENSOR = "blk.0.attn_output.weight"


def build():
    """The measured original attention-output Q4_0 matvec operator."""
    return Q4MatvecOp(
        in_dim=IN_DIM,
        out_dim=OUT_DIM,
        tensor=TENSOR,
        representative_rows=256,
        view="attention-output-family",
    )


if __name__ == "__main__":
    import json

    print(json.dumps(build().to_json(), indent=2))
