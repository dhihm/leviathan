"""User op model: the dominant Gemma 4 RMSNorm family (phase 10 example).

Describes the hidden-size (n = 2816) scaled RMSNorm that every layer's
attention/FFN norm sites and the final norm use, bound to the layer-0
attn_norm weight for representative bytes. The Swift side re-validates the
declared shape against the actual GGUF tensor ([2816] F32) and rejects a
mismatch; the eps comes from the GGUF config (rmsNormEps 1e-6).

`vectors` is the CAPTURE-TIME component batch (the representative vector
count the component gates evaluate); the runtime dispatch is per-call —
production calls this family with vectors=1, and the installed generated
kernel handles whatever per-call vector count rmsnorm() passes.
"""


def build():
    from rms_op_model import RMSNormOp
    return RMSNormOp(
        n=2816,
        vectors=8,
        scaled=True,
        eps=1e-6,
        tensor="blk.0.attn_norm.weight",
        representative_vectors=8,
    )
