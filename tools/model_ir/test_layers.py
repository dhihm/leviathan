"""The functional layer wrappers equal the Hugging Face layers (small random config, float32)."""
import sys
import unittest

import torch

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)

from ornith_layers import AttnDecoderLayer, GdnDecoderLayer


def small_config():
    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    )
    cfg._attn_implementation = "eager"
    return cfg


def randomize(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)


def rope_tables(cfg, t, start=0):
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    pos = torch.arange(start, start + t)[None]
    probe = torch.zeros(1, t, cfg.hidden_size)
    try:
        cos, sin = rot(probe, pos)
    except Exception:
        cos, sin = rot(probe, pos[None].expand(3, 1, t))
    return cos[0], sin[0]  # [T, rd]


class GdnLayerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = small_config()
        self.hf = Qwen3_5DecoderLayer(self.cfg, 0).eval()
        randomize(self.hf, 1)
        self.fn = GdnDecoderLayer(self.hf).eval()
        m = self.hf.linear_attn
        self.conv0 = torch.zeros(m.conv_kernel_size - 1, m.conv_dim)
        self.rec0 = torch.zeros(m.num_v_heads, m.head_k_dim, m.head_v_dim)

    def reference(self, x):
        cos, sin = rope_tables(self.cfg, x.shape[0])
        with torch.no_grad():
            return self.hf(x[None], position_embeddings=(cos[None], sin[None]))[0]

    def test_whole_sequence(self):
        x = torch.randn(11, self.cfg.hidden_size, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            y, _, _ = self.fn(x, self.conv0, self.rec0)
        torch.testing.assert_close(y, self.reference(x), rtol=1e-4, atol=1e-4)

    def test_token_by_token_equals_whole_sequence(self):
        x = torch.randn(9, self.cfg.hidden_size, generator=torch.Generator().manual_seed(3))
        conv, rec = self.conv0, self.rec0
        outs = []
        with torch.no_grad():
            for t in range(x.shape[0]):
                y, conv, rec = self.fn(x[t : t + 1], conv, rec)
                outs.append(y)
        torch.testing.assert_close(torch.cat(outs), self.reference(x), rtol=1e-4, atol=1e-4)

    def test_split_sequence_carries_state(self):
        x = torch.randn(10, self.cfg.hidden_size, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            y1, conv, rec = self.fn(x[:6], self.conv0, self.rec0)
            y2, _, _ = self.fn(x[6:], conv, rec)
        torch.testing.assert_close(torch.cat([y1, y2]), self.reference(x), rtol=1e-4, atol=1e-4)


class AttnLayerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = small_config()
        self.hf = Qwen3_5DecoderLayer(self.cfg, 1).eval()
        randomize(self.hf, 5)
        self.fn = AttnDecoderLayer(self.hf).eval()
        a = self.hf.self_attn
        self.kc = torch.zeros(0, self.cfg.num_key_value_heads, a.head_dim)

    def reference(self, x):
        t = x.shape[0]
        cos, sin = rope_tables(self.cfg, t)
        mask = torch.full((t, t), float("-inf")).triu(1)[None, None]
        with torch.no_grad():
            return self.hf(x[None], position_embeddings=(cos[None], sin[None]), attention_mask=mask)[0]

    def test_whole_sequence(self):
        x = torch.randn(7, self.cfg.hidden_size, generator=torch.Generator().manual_seed(6))
        cos, sin = rope_tables(self.cfg, 7)
        with torch.no_grad():
            y, _, _ = self.fn(x, cos, sin, self.kc, self.kc.clone())
        torch.testing.assert_close(y, self.reference(x), rtol=1e-4, atol=1e-4)

    def test_incremental_with_cache(self):
        x = torch.randn(8, self.cfg.hidden_size, generator=torch.Generator().manual_seed(7))
        kc, vc = self.kc, self.kc.clone()
        outs = []
        with torch.no_grad():
            for t in range(x.shape[0]):
                cos, sin = rope_tables(self.cfg, 1, start=t)
                y, k, v = self.fn(x[t : t + 1], cos, sin, kc, vc)
                kc, vc = torch.cat([kc, k]), torch.cat([vc, v])
                outs.append(y)
        torch.testing.assert_close(torch.cat(outs), self.reference(x), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
