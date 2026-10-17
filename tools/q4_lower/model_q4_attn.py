"""User op model: the dominant Gemma 4 Q4_0 matvec family (phase 9 example).

Describes the layer-0 attention-Q projection geometry of Gemma 4 26B-A4B
exactly as the GGUF tensor directory pins it ([2816, 4096] = inDim 2816
hidden, outDim 4096 = 16 heads x 256 local head dim), with the real tensor
binding for representative weight bytes. The Swift side re-validates the
declared shape against the actual tensor and rejects a mismatch.
"""


def build():
    from q4_op_model import Q4MatvecOp
    return Q4MatvecOp(
        in_dim=2816,
        out_dim=4096,
        tensor="blk.0.attn_q.weight",
        representative_rows=256,
    )
