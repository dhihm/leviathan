//! Model IR: the graph exported from torch.export (tools/model_ir), its loader and validator, and
//! a CPU reference interpreter that defines the semantics of every operator.

pub mod bind;
pub mod interp;
pub mod ir;
pub mod json;
pub mod pack;
pub mod pass;
pub mod tensor;
