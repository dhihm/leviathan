"""The exported graphs keep the semantics: they run at several token counts and equal Hugging Face."""
import unittest
from collections import Counter

import torch
from torch.export import Dim, export

from test_layers import AttnLayerTest, GdnLayerTest, rope_tables, small_config  # noqa: F401
from ornith_layers import AttnDecoderLayer, GdnDecoderLayer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
from test_layers import randomize


def custom_ops(ep):
    return [str(n.target) for n in ep.graph.nodes if n.op == "call_function" and str(n.target).startswith("lev.")]


class ExportedGraphTest(unittest.TestCase):
    def test_gated_delta_layer_dynamic_tokens(self):
        cfg = small_config()
        hf = Qwen3_5DecoderLayer(cfg, 0).eval()
        randomize(hf, 11)
        fn = GdnDecoderLayer(hf).eval()
        m = hf.linear_attn
        conv0 = torch.zeros(m.conv_kernel_size - 1, m.conv_dim)
        rec0 = torch.zeros(m.num_v_heads, m.head_k_dim, m.head_v_dim)
        t = Dim("T", min=1, max=64)
        ep = export(fn, (torch.randn(4, cfg.hidden_size), conv0, rec0), dynamic_shapes={"x": {0: t}, "conv_state": None, "rec_state": None})
        self.assertEqual(
            Counter(custom_ops(ep)),
            Counter({"lev.rmsnorm.default": 2, "lev.causal_conv1d_state.default": 1, "lev.gated_delta_rule.default": 1, "lev.rmsnorm_gated.default": 1, "lev.swiglu.default": 1}),
        )
        g = ep.module()
        for n in (1, 3, 9, 17):
            x = torch.randn(n, cfg.hidden_size, generator=torch.Generator().manual_seed(n))
            with torch.no_grad():
                want = fn(x, conv0, rec0)
                got = g(x, conv0, rec0)
            for a, b in zip(want, got):
                torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)

    def test_attention_layer_dynamic_tokens_and_cache(self):
        cfg = small_config()
        hf = Qwen3_5DecoderLayer(cfg, 1).eval()
        randomize(hf, 12)
        fn = AttnDecoderLayer(hf).eval()
        a = hf.self_attn
        hkv, hd = cfg.num_key_value_heads, a.head_dim
        rd = rope_tables(cfg, 2)[0].shape[-1]
        t, s = Dim("T", min=1, max=64), Dim("S", min=0, max=256)
        ex = (torch.randn(3, cfg.hidden_size), torch.randn(3, rd), torch.randn(3, rd), torch.randn(2, hkv, hd), torch.randn(2, hkv, hd))
        ep = export(fn, ex, dynamic_shapes={"x": {0: t}, "cos": {0: t}, "sin": {0: t}, "k_cache": {0: s}, "v_cache": {0: s}})
        self.assertEqual(
            Counter(custom_ops(ep)),
            Counter({"lev.rmsnorm.default": 4, "lev.rope.default": 2, "lev.causal_attention.default": 1, "lev.swiglu.default": 1}),
        )
        g = ep.module()
        for n, past in ((1, 0), (4, 0), (1, 7), (5, 3)):
            x = torch.randn(n, cfg.hidden_size, generator=torch.Generator().manual_seed(n + past))
            cos, sin = rope_tables(cfg, n, start=past)
            kc = torch.randn(past, hkv, hd, generator=torch.Generator().manual_seed(past + 1))
            vc = torch.randn(past, hkv, hd, generator=torch.Generator().manual_seed(past + 2))
            with torch.no_grad():
                want = fn(x, cos, sin, kc, vc)
                got = g(x, cos, sin, kc, vc)
            for w, o in zip(want, got):
                torch.testing.assert_close(o, w, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()