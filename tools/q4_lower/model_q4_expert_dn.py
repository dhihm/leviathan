"""User op model: the DOMINANT 3-D down expert family (phase 11).

Describes ONE expert's [outDim, inDim] slice of the layer-0
ffn_down_exps tensor exactly as the GGUF pins the family
([704, 2816, 128] = inDim 704 expert intermediate, outDim 2816
hidden, 128 experts) with the expert-row-slice view. The Swift side
re-validates the declared slice shape against the actual 3-D tensor
(dims[0]=inDim, dims[1]=outDim, dims[2]=experts) and rejects a
mismatch; the family binding serves every layer's same-shaped slices.
"""


def build():
    from q4_op_model import Q4MatvecOp
    return Q4MatvecOp(
        in_dim=704,
        out_dim=2816,
        tensor="blk.0.ffn_down_exps.weight",
        representative_rows=256,
        view="expert-row-slice",
    )
