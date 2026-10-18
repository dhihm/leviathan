# model_ir: torch.export capture of the Ornith (Qwen3.5) layers

Build-time Python side of the Forge pipeline for the Vulkan backend: the model
is captured with `torch.export` into a graph whose token mixers are custom
operators (`lev.causal_conv1d_state`, `lev.gated_delta_rule`,
`lev.causal_attention`), so a later stage can lower and optimize them as units.

- `lev_ops.py`: the custom operators with CPU reference semantics and fake
  (meta) kernels. State is functional (in and out), nothing is mutated.
- `ornith_layers.py`: export-friendly wrappers of the Hugging Face decoder
  layers (same parameters, flat `[T, D]` tokens, explicit state).
- `export_layers.py`: exports both layer types at real size on the meta device
  (no weights needed) with a symbolic token count.
- `ornith_model.py`: the whole model (embedding, 32 layers, final norm, output projection) with every
  recurrent state and KV cache as an explicit input and output.
- `ir_export.py`: serializes the `torch.export` result of the whole model into the model IR (JSON,
  format `lev-model-ir-1`; no weights, parameters keep their Hugging Face names). The Rust crate
  `rust/crates/lev-ir` loads it, validates it and runs it on the CPU as the reference interpreter.
- `make_fixture.py`: tiny random model, its IR, weights, inputs and PyTorch outputs, used by the
  Rust tests (`rust/crates/lev-ir/tests/fixtures/tiny`).
- `test_layers.py`, `test_export.py`: the wrappers and the exported graphs equal
  the Hugging Face layers (small random config, several token counts).

Python 3.12 with torch (CPU build) and transformers; any virtual environment:

    python -m unittest test_layers test_export
    python export_layers.py [nodes.json]

The Rust runtime bundles the export of the whole model
(`rust/crates/lev-model/ir/ornith-1.5-9b.json`) and lowers it at start-up; regenerate it with

    python ir_export.py ../../rust/crates/lev-model/ir/ornith-1.5-9b.json

after changing the operators or the layer wrappers (the Rust tests pin the lowered operation counts).

`make_fixture_ragged.py` and `--ragged` give the multi-sequence layout (`ragged-v1`: per-token slot and position, per-layer state pools); the Rust runtime bundles it as `ornith-1.5-9b-ragged.json`.

`make_fixture_paged.py` and `--paged` give the paged-KV layout (`paged-v1`: block pools, block table, per-token pool position), bundled as `ornith-1.5-9b-paged.json`.

Status: the single-sequence, ragged slab and paged layouts work. The multi-sequence flat layout, paged KV
attention and state slots are not implemented yet.