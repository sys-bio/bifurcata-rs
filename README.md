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
| M2 | Newton, pseudo-arclength continuation, equilibria, fold/branch-point/Hopf detection, JSON output, Antimony models | Done |
| M3 | Normal forms, branch switching | Next |
| M4 | Codimension-2 fold and Hopf curves | — |

## Building

```
cargo test
cargo check --target wasm32-unknown-unknown
cargo test --features antimony      # Antimony models via websim's model crate
```

The baseline tests compare runs with the Delphi version's stored results, read
in place from the Delphi project (set `BIFURCATA_DELPHI_DIR`); they skip when
it is absent.

## The console harness

```
cargo run --features antimony --bin bifurcata -- run model.ant --param k1 --range 0:2 -o out.json
cargo run --bin bifurcata -- run brusselator                 # a built-in model
cargo run --bin bifurcata -- compare out.json baseline.json --locations-only
cargo run --bin bifurcata -- models
```

`run` continues an equilibrium branch in both directions from a steady state
(or `--start v1,v2,...`), detects folds, branch points, Hopf points and neutral
saddles, and writes the `bifurcata/1` JSON of the Delphi version (or
`--format csv`). Exit codes: 0 done, 1 stopped at a bound, 2 continuation
failed, 3 no start point, 4 model error, 5 `compare` mismatch.

## Using the library

```rust
use bifurcata::continuation::{ContinuationEngine, ContinuationOptions};
use bifurcata::equilibrium::EquilibriumCurve;
use bifurcata::test_problems::Brusselator;

let options = ContinuationOptions { parameter_min: 1.0, parameter_max: 3.0, ..Default::default() };
let curve = EquilibriumCurve::new(Brusselator, &[1.0, 1.2], 1, &options)?;
let x0 = curve.pack(&Brusselator::equilibrium(1.0, 1.2), 1.2);
let mut engine = ContinuationEngine::new(curve, options);
engine.initialise(&x0, 1)?;
while engine.step() == bifurcata::types::StepResult::Ok {
    // draw engine.branch().last_point() as the branch grows
}
```
