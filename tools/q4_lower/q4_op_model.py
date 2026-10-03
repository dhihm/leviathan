"""Q4_0 matvec operation model for LeviathanForge operator lowering (phase 9).

A user op model (.py) describes ONE bounded quantized operator with full
semantics/shape/layout/quant/precision metadata via Q4MatvecOp. The capture
script (export_q4_op.py) executes the user's file and emits op JSON; the
Swift side re-validates every field and rejects anything unsupported.

This is an OPERATOR-level input (one Q4_0 matrix-vector product family) —
not a claim of an arbitrary-Python-model compiler.
"""

LAYOUTS = ["row-major-q4_0-blocks"]
QUANTS = ["q4_0"]
ACCUMULATES = ["f32"]
INPUT_PRECISIONS = ["f32"]
VIEWS = [None, "expert-row-slice"]

EXPERT_FAMILIES = [".ffn_gate_up_exps.weight", ".ffn_down_exps.weight"]


class Q4MatvecOp:
    """y[row] = sum_col W[row, col] * x[col]; W row-major Q4_0 blocks.

    view=None: W is the bound 2-D tensor itself.
    view="expert-row-slice": the bound tensor is a 3-D expert family
    [inDim, outDim, experts]; W is ONE expert's contiguous row-major
    [outDim, inDim] slice (the runtime slice byte offset addresses the
    expert; every expert of the family shares the same slice shape).
    """

    def __init__(self, in_dim, out_dim, tensor=None, representative_rows=None,
                 view=None):
        if not isinstance(in_dim, int) or not isinstance(out_dim, int):
            raise TypeError("in_dim/out_dim must be ints")
        if in_dim <= 0 or out_dim <= 0 or in_dim % 32 != 0:
            raise ValueError(
                f"bad shape in_dim={in_dim} out_dim={out_dim} (in_dim % 32 == 0 required)")
        if tensor is not None and not isinstance(tensor, str):
            raise TypeError("tensor must be a string or None")
        if representative_rows is not None:
            if not isinstance(representative_rows, int) or representative_rows <= 0:
                raise ValueError("representative_rows must be a positive int")
            if representative_rows > out_dim:
                raise ValueError("representative_rows exceeds out_dim")
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
        if view == "expert-row-slice":
            if tensor is None:
                raise ValueError("expert-row-slice requires a tensor binding")
            if not any(tensor.endswith(f) for f in EXPERT_FAMILIES):
                raise ValueError(
                    "expert-row-slice binds a blk.<L>.ffn_{gate_up,down}_exps.weight "
                    f"tensor, got {tensor!r}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tensor = tensor
        self.representative_rows = representative_rows
        self.view = view

    def to_json(self):
        return {
            "schema": 1,
            "op": "q4_0.matvec",
            "semantics": "dot(W_row, x)",
            "inDim": self.in_dim,
            "outDim": self.out_dim,
            "layout": LAYOUTS[0],
            "quant": QUANTS[0],
            "accumulate": ACCUMULATES[0],
            "inputPrecision": INPUT_PRECISIONS[0],
            "tensor": self.tensor,
            "representativeRows": self.representative_rows,
            "view": self.view,
        }


def check_op(obj):
    """Structural validation mirrored on the Swift side (fail early here too)."""
    required = {"schema": 1, "op": "q4_0.matvec", "semantics": "dot(W_row, x)",
                "layout": LAYOUTS[0], "quant": QUANTS[0], "accumulate": ACCUMULATES[0],
                "inputPrecision": INPUT_PRECISIONS[0]}
    for key, want in required.items():
        if obj.get(key) != want:
            raise ValueError(f"field {key!r} must be {want!r}, got {obj.get(key)!r}")
    for key in ("inDim", "outDim"):
        v = obj.get(key)
        if not isinstance(v, int) or v <= 0 or v % 1 != 0:
            raise ValueError(f"field {key!r} must be a positive int")
    if obj["inDim"] % 32 != 0:
        raise ValueError("inDim must be a multiple of the 32-value Q4_0 block")
    rows = obj.get("representativeRows")
    if rows is not None and (not isinstance(rows, int) or not 0 < rows <= obj["outDim"]):
        raise ValueError("representativeRows must satisfy 0 < rows <= outDim")
    view = obj.get("view")
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    # Re-run the constructor checks (tensor/view consistency included).
    Q4MatvecOp(obj["inDim"], obj["outDim"], tensor=obj.get("tensor"),
               representative_rows=rows, view=view)
    return obj
