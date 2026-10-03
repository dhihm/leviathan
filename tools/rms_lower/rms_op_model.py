"""RMSNorm operation model for LeviathanForge operator lowering (phase 10).

A user op model (.py) describes ONE bounded RMSNorm family with full
semantics/shape/layout/quant/precision metadata via RMSNormOp. The capture
script (export_rms_op.py) executes the user's file and emits op JSON; the
Swift side re-validates every field (including the bound tensor's actual
GGUF geometry) and rejects anything unsupported.

This is an OPERATOR-level input (one RMSNorm shape family) — not a claim of
an arbitrary-Python-model compiler.
"""

LAYOUTS = ["row-major-f32-vectors"]
QUANTS = ["f32"]
REDUCTIONS = ["serial-order-i0..n"]
INPUT_PRECISIONS = ["f32"]


class RMSNormOp:
    """y[i] = x[i] * inv_rms(x) * w[i] over vectors of n float32 values.

    inv_rms(x) = 1 / sqrt(mean(x*x) + eps), the sum accumulated serially in
    the original element order (i = 0..n-1) — the incumbent kernel's exact
    reduction, which the generated lowering must preserve bit-for-bit.
    """

    def __init__(self, n, vectors, scaled, eps, tensor=None,
                 representative_vectors=None):
        if not isinstance(n, int) or not isinstance(vectors, int):
            raise TypeError("n/vectors must be ints")
        if n <= 0 or n % 4 != 0:
            raise ValueError(f"bad n={n} (positive, multiple of 4 required)")
        if vectors <= 0:
            raise ValueError(f"bad vectors={vectors}")
        if not isinstance(scaled, bool):
            raise TypeError("scaled must be a bool")
        if not isinstance(eps, (int, float)) or not (0 <= eps <= 1e-3):
            raise ValueError(f"bad eps={eps}")
        if tensor is not None and not isinstance(tensor, str):
            raise TypeError("tensor must be a string or None")
        if scaled and tensor is None:
            raise ValueError("a scaled op must bind a weight tensor")
        if representative_vectors is not None:
            if not isinstance(representative_vectors, int) or representative_vectors <= 0:
                raise ValueError("representative_vectors must be a positive int")
            if representative_vectors > vectors:
                raise ValueError("representative_vectors exceeds vectors")
        self.n = n
        self.vectors = vectors
        self.scaled = scaled
        self.eps = float(eps)
        self.tensor = tensor
        self.representative_vectors = representative_vectors

    def to_json(self):
        return {
            "schema": 1,
            "op": "rmsnorm",
            "semantics": "y[i] = x[i] * inv_rms(x) * w[i]",
            "n": self.n,
            "vectors": self.vectors,
            "scaled": self.scaled,
            "eps": self.eps,
            "layout": LAYOUTS[0],
            "quant": QUANTS[0],
            "reduction": REDUCTIONS[0],
            "inputPrecision": INPUT_PRECISIONS[0],
            "tensor": self.tensor,
            "representativeVectors": self.representative_vectors,
        }


def check_op(obj):
    """Structural validation mirrored on the Swift side (fail early here too)."""
    required = {"schema": 1, "op": "rmsnorm",
                "semantics": "y[i] = x[i] * inv_rms(x) * w[i]",
                "layout": LAYOUTS[0], "quant": QUANTS[0],
                "reduction": REDUCTIONS[0], "inputPrecision": INPUT_PRECISIONS[0]}
    for key, want in required.items():
        if obj.get(key) != want:
            raise ValueError(f"field {key!r} must be {want!r}, got {obj.get(key)!r}")
    for key in ("n", "vectors"):
        v = obj.get(key)
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"field {key!r} must be a positive int")
    if obj["n"] % 4 != 0:
        raise ValueError("n must be a multiple of 4")
    if not isinstance(obj.get("scaled"), bool):
        raise ValueError("field 'scaled' must be a bool")
    if obj["scaled"] and obj.get("tensor") is None:
        raise ValueError("a scaled op must bind a weight tensor")
    eps = obj.get("eps")
    if not isinstance(eps, (int, float)) or not (0 <= eps <= 1e-3):
        raise ValueError("field 'eps' must be a float in [0, 1e-3]")
    reps = obj.get("representativeVectors")
    if reps is not None and (not isinstance(reps, int) or not 0 < reps <= obj["vectors"]):
        raise ValueError("representativeVectors must satisfy 0 < reps <= vectors")
    return obj
