# Hardware-derived Q4 SIMD lowering

Leviathan's bounded Q4 operator Forge now derives an additional plain SIMD lane mapping from a validated operator shape and actual pipeline hardware. One 32-lane SIMD group computes a row; each lane owns one quantization component per Q4_0 block and traverses all blocks. A SIMD reduction writes the result. Legal row groups are 1, 4 and 8, with actual execution width exactly 32 and the compiled threadgroup cap checked. Candidate generation, compilation, numerical gates and local measurement run without an agent. Existing candidates remain available; the maximum search set grows from 13 to 16.

This is an original operator lowering extension, not arbitrary whole-model Python compilation. The separate Torch-export static DAG path remains bounded to its documented supported operators.

## Python input and routing

`tools/q4_lower/model_q4_attn_output.py` supplies the original operator DSL input for `blk.0.attn_output.weight`, shape 4096 to 2816. The explicit `attention-output-family` view can serve strict `blk.<ASCII digits>.attn_output.weight` siblings only at exactly the same input/output shape. Global attention output matrices with a different input dimension use the incumbent path. A missing/wrong family name, quantization type or bound GGUF shape rejects before artifact creation. No general all-Q4 override or default runtime installation is introduced.

Code generation is `q4gen-v3`. Previous Q4 artifacts reject stale codegen identity; changes to core sources also invalidate source-bound DAG artifacts. Re-forging is explicit. Component-only artifacts remain non-installable in the production runner. Persisted model artifacts still require the existing full-logit, finite-value, greedy-token, dispatch, hardware, source, binary and model identity gates. The full-model selector's correctness tolerances and policy are unchanged.

## Component measurements

On the Apple M1 Max, two existing real Q4 tensors were read in bounded slices: `blk.0.attn_q.weight` (2816 to 4096) and `blk.0.attn_output.weight` (4096 to 2816), 6,488,064 bytes each. The token embedding is Q6_K and is explicitly unsupported by this Q4 experiment. No model was downloaded or fully dequantized.

The standalone experiment compares the actual incumbent source at threadgroup width 64 with original plain/default-fast-math and compensated/no-fast-math SIMD variants. Every dispatch NaN-fills the output, requires completion and positive GPU timestamps, and checks every row finite. Independent FP64 references cover the first 256 rows for three deterministic signed, alternating and one-hot inputs. Each variant/input uses five warmups and fifteen timed samples with interleaved order. GPU and complete submit/wait/readback times are recorded separately. Numerical tolerance is unchanged at absolute 1e-4 plus relative 1e-4 times reference magnitude.

Three fresh processes each passed 10,752 FP64 comparisons. Plain R8 was selected for attention output in all three, with geometric-mean GPU ratios 0.70098, 0.66653 and 0.71183 versus the width-64 incumbent (28.8 to 33.3 percent reductions). Query projection fell back in two runs and selected plain R8 in one; its improvement is not stable enough for a universal claim. Compensated variants were accurate but costlier. This is component evidence, not an end-to-end decode improvement. Shape-specific measured selection is essential.

## Fresh Ollama reference

A separate test-owned localhost daemon used the existing same-digest GGUF registration, one slot and five warm raw-Hello rounds, each preceded by a distractor. All five cache-policy and deterministic text/count checks passed. The non-BOS prompt token was freshly evaluated; one BOS prefix token was reused. Median server-standard decode speed was 58.078 tokens/s; alternate first-to-last normalization was 56.263 tokens/s. These server counters differ in scope from Leviathan's synchronous adapter wall clock. Buffered HTTP chunks are not token interval evidence. The user's existing daemon was not changed; only the temporary test daemon/model was cleaned up. This reference does not establish Leviathan parity or superiority.

## Validation and reproduction

The integrated Swift regression suite passed 810 checks, including actual ragged 35-row SIMD output versus FP64, hardware legality, strict family routing and source tamper rejection. The generic persistent DAG runtime passed all 22 acceptance cases on freshly bound artifacts. Independent actual-GGUF input/output shape changes rejected without writing an artifact. All implementation and test changes were externally authored through configured GLM-5.3/ZCode and mechanically applied; documentation and measurement orchestration were written locally. No external project source or history was supplied.

```sh
swift build -c release
python3 tools/q4_lower/export_q4_op.py \
  tools/q4_lower/model_q4_attn_output.py NEW_OP.json
swiftc -O -I .build/release/Modules \
  tools/q4_simd_qualification/main.swift \
  .build/release/LeviathanCore.build/*.swift.o -o .build/release/q4-simd-qualify
.build/release/q4-simd-qualify --gguf MODEL.gguf --out NEW_COMPONENT_REPORT.json
.build/release/leviathan-cli q4-lower select \
  --op NEW_OP.json --gguf MODEL.gguf --out NEW_MODEL_ARTIFACT.json \
  --gen-tokens 32 --timed-tokens 31
```

The standalone report refuses to overwrite an existing output. The full-model command performs component screening, full numerical/greedy replay, paired timing, selection, artifact reload and verified execution. A successful component report does not bypass any model gate.

Full-model qualification is pending at the time of this initial local integration record; no full-model SIMD gain is claimed. The goal of closing the Ollama gap remains open.

[Component round evidence](evidence/q4-simd-forge/component-rounds.json) and [fresh Ollama counter evidence](evidence/q4-simd-forge/ollama-current.json) contain no weights, binaries, provider credentials or external-author conversation identifiers.
