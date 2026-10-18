//! The IR interpreter reproduces PyTorch on a tiny Ornith-shaped model (four layers: three
//! linear attention, one full attention) with non-empty recurrent states and KV cache.
//! Regenerate the fixture with: python tools/model_ir/make_fixture.py <dir>

use std::path::Path;

use lev_ir::interp::{self, V};
use lev_ir::ir::Graph;
use lev_ir::pack::{self, Packed};

fn fixture(dir: &str, name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(dir).join(name)
}

#[test]
fn tiny_model_matches_pytorch() {
    check("tiny");
}

#[test]
fn tiny_ragged_batch_matches_pytorch() {
    check("tiny_ragged");
}

fn check(dir: &str) {
    let g = Graph::parse(&std::fs::read_to_string(fixture(dir, "ir.json")).unwrap()).unwrap();
    g.validate().unwrap();
    for (op, _) in g.op_counts() {
        assert!(interp::is_supported(&op), "unsupported operator {op}");
    }
    let params = pack::read(&std::fs::read(fixture(dir, "weights.pack")).unwrap()).unwrap();
    let inputs = pack::read(&std::fs::read(fixture(dir, "inputs.pack")).unwrap()).unwrap();
    let expected = pack::read(&std::fs::read(fixture(dir, "expected.pack")).unwrap()).unwrap();
    let outs = interp::run(&g, &params, &inputs).unwrap();
    assert_eq!(outs.len(), expected.len());
    for (i, o) in outs.iter().enumerate() {
        let Some(Packed::F32(want)) = expected.get(&format!("out_{i}")) else { panic!("missing out_{i}") };
        let V::T(got) = &**o else { panic!("output {i} is not a tensor") };
        assert_eq!(got.shape, want.shape, "out_{i} shape");
        let max_ref = want.data.iter().fold(0f32, |m, v| m.max(v.abs()));
        let max_abs = got.data.iter().zip(&want.data).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(max_abs <= 1e-4 + 1e-4 * max_ref, "out_{i}: max abs error {max_abs} (max |ref| {max_ref})");
    }
}

#[test]
fn validation_rejects_a_forward_reference() {
    let text = r#"{"format":"lev-model-ir-1","inputs":[],"params":[],"nodes":[{"op":"aten::neg.default","args":[{"n":1}],"kwargs":{},"out":null},{"op":"aten::neg.default","args":[{"n":0}],"kwargs":{},"out":null}],"outputs":[{"n":1}]}"#;
    let g = Graph::parse(text).unwrap();
    assert!(g.validate().is_err());
}
