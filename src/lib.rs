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
//! Milestones M1–M2: the foundations (types, complex arithmetic, dense linear
//! algebra, the bordered solver, the problem contract, test problems), then
//! Newton, pseudo-arclength continuation of equilibria with fold, branch-point
//! and Hopf detection, the `bifurcata/1` output schema and its comparator, and
//! (feature `antimony`) Antimony models through websim's model crate.

#[cfg(feature = "antimony")]
pub mod antimony;
pub mod bialternate;
pub mod bordered;
pub mod complex;
pub mod continuation;
pub mod equilibrium;
pub mod linalg;
pub mod matrix;
pub mod models;
pub mod newton;
pub mod problem;
pub mod run;
pub mod runspec;
pub mod serialise;
pub mod test_problems;
pub mod types;
