# bifurcata

Numerical bifurcation analysis of parameterised ODE systems `du/dt = F(u, λ)`,
in pure Rust: continuation of equilibria, detection and classification of
local bifurcations (folds, branch points, Hopf points), normal forms, branch
switching and codimension-2 curves.

A rewrite of the Delphi library **Bifurcata**, written from its specification
and checked against its test suite and stored results. Being pure Rust, it runs
in the browser (WebAssembly) as well as on the desktop; it is used by
[websim](https://github.com/sys-bio/websim)
([live](https://sys-bio.github.io/websim/)), whose model layer supplies the
Antimony models and conservation analysis.

## Status

Work in progress, following the plan in websim's
[`docs/bifurcation-rust-plan.md`](https://github.com/sys-bio/websim/blob/main/docs/bifurcation-rust-plan.md).

| Milestone | Content | Status |
|---|---|---|
| M1 | Types, complex arithmetic, dense linear algebra (faer), bordered solver, problem trait, test problems | Done |
| M2 | Newton, pseudo-arclength continuation, equilibria, fold/branch-point/Hopf detection, JSON output | Next |
| M3 | Normal forms, branch switching | — |
| M4 | Codimension-2 fold and Hopf curves | — |

## Building

```
cargo test
cargo check --target wasm32-unknown-unknown
```
