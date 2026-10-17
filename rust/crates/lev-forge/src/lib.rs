//! Forge for the cross-platform engine: schedule derivation and kernel source
//! emitters. Pure functions only; device access lives in the backend crates.

pub mod coopmat;
pub mod glsl;
pub mod ops;
pub mod q8;
pub mod q6k;
pub mod schedule;
pub mod sha256;
