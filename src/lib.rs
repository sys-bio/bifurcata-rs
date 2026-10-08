//! # bifurcata
//!
//! Numerical bifurcation analysis of parameterised ODE systems
//! `du/dt = F(u, λ)`: continuation of equilibria, detection and classification
//! of local bifurcations, and branch switching.
//!
//! A pure-Rust rewrite of the Delphi library Bifurcata, written from its
//! specification (`BifurcationSpec.md` in the Delphi project, the authority on
//! what the library does) and checked against its tests and baselines. Pure
//! Rust, so that it runs in the browser as well as on the desktop.
//!
//! This is milestone M1: the foundations — types, complex arithmetic, dense
//! linear algebra, the bordered solver, the problem contract and test problems.

pub mod bordered;
pub mod complex;
pub mod linalg;
pub mod matrix;
pub mod problem;
pub mod test_problems;
pub mod types;
