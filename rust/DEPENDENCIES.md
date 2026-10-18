# Rust workspace dependencies

Policy: third-party crates are allowed in this workspace only (the Swift core
stays dependency-free). Every third-party crate is listed here with its
reason; versions are pinned by the committed `Cargo.lock`.

| Crate | Version | Used by | Reason |
| --- | --- | --- | --- |
| `ash` | `=0.38.0` (Vulkan 1.3.281 bindings), features `std`, `loaded` only | `lev-vulkan` | Thin, unsafe 1:1 Vulkan bindings. Replaces hand-written FFI declarations for the Vulkan C API; no abstraction layer, so the schedule and kernel decisions stay in this repository. `loaded` loads the system Vulkan loader at run time (no link-time SDK requirement). |

Transitive dependencies pulled in by `ash` (pinned in `Cargo.lock`):
`libloading` 0.8.9 (loads the Vulkan loader library), `cfg-if` 1.0.5,
`windows-link` 0.2.1 (Windows import declarations used by `libloading`).

`lev-gguf`, `lev-cpu`, `lev-forge`, `lev-ir` and `lev-tokenizer` use the standard library only (`lev-ir` carries its own small JSON parser instead of `serde_json`), and
`lev-cli` depends only on the workspace crates. Shader compilation currently shells out to `glslc` from the Vulkan SDK
(not a Cargo dependency); see the open decision between in-process
compilers before the emitter is used at user run time.
