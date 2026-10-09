# bifurcata

Numerical bifurcation analysis of ODE models `du/dt = F(u, λ)` in pure Rust.
Today it follows **branches of equilibria** as a parameter varies, works out
their **stability** at every point, and **detects and locates folds, branch
points and Hopf points** along the way. It reads biochemical models written
in Antimony, and because it is pure Rust it runs in the browser as well as on
the desktop.

It is a rewrite of the Delphi library **Bifurcata**, written from that
library's specification and checked against its test suite and its stored
results. It powers the Bifurcation tab of
[websim](https://github.com/sys-bio/websim) — **try it at
https://sys-bio.github.io/websim/** (Examples → Bifurcation examples, then Run).

## What it can do now

- **Equilibrium continuation** by pseudo-arclength (PALC), so a branch is
  followed round folds rather than stopping at them, with a Moore–Penrose
  corrector as an alternative. Step sizes adapt to the corrector's effort,
  and a step whose tangent turns too sharply is rejected, so the curve cannot
  jump to a neighbouring branch.
- **Stability** at every point: the full spectrum of the Jacobian, and the
  number of unstable directions.
- **Detection and location** of
  - **folds** (limit points, saddle-nodes),
  - **branch points**, where two branches of equilibria cross,
  - **Hopf points**, with their frequency ω, told apart from **neutral
    saddles** (a real pair of eigenvalues summing to zero, which zeroes the
    same test function but is not a bifurcation).

  Each is located by re-walking the bracketing step in short continuation
  steps, then bisecting, which locates them to about eight significant
  figures even at large step sizes.
- **Antimony models** (feature `antimony`), through websim's model layer:
  conservation laws are found automatically and the analysis runs on the
  reduced system, and conserved totals can themselves be varied (as
  `_CSUM0`, `_CSUM1`, …). A model can carry its own run settings in a
  `[bifurcation]` comment block.
- **Built for interactive use.** The continuation engine is an iterator: each
  call to `step()` adds one point, so a user interface can draw the branch as
  it grows and stop it at any time. Callbacks report every point, every
  corrector attempt (rejected ones included) and every candidate the analysis
  declined, with its reason.
- **Output** in the `bifurcata/1` JSON schema shared with the Delphi version
  (an exact round trip), CSV, and a comparator for regression testing.
- **A console program**, `bifurcata`, with `run`, `compare` and `models`.

### Not yet

Next (milestone M3): **normal forms** — classifying a Hopf point as
supercritical (a stable limit cycle is born) or subcritical, and testing
whether a fold is non-degenerate — and **branch switching**, to continue the
other branch through a branch point. After that: codimension-2 fold and Hopf
curves (M4). Periodic orbits are a later decision.

## How well it agrees

- **The Delphi version's stored results**: 24 runs (4 built-in models, 19
  Antimony models, one continued in a conserved total) agree in the kinds,
  number and locations of their bifurcations to six significant figures,
  and in every sampled point of every branch to 1e-6.
- **Closed forms**: AUTO's PP2 demo (branch points at 0.6 and
  0.821904345374, a fold at 0.832929322234, a Hopf at 0.671593847479, a
  neutral saddle at 0.4) and MatCont's Lab 2 branch point
  (d = −0.3130208333) to 1e-8; the n-stage Goodwin oscillator's Hopf at
  ω = tan(π/n) for n = 6, 10 and 20.
- **Published values**: the catalytic oscillator of MatCont's manual (§8.1.5)
  to six or seven figures; the Tyson–Novak 2001 cell-cycle model's two folds
  and three Hopf points.
- The Delphi test suites for Newton, continuation, the bialternate product,
  scaling and serialisation are ported, and each test has been checked by
  breaking the code on purpose and watching it fail.

## Quick start

### From the command line

```
cargo run --features antimony --bin bifurcata -- run model.ant --param k1 --range 0:2 -o out.json
cargo run --bin bifurcata -- run brusselator          # a built-in model
cargo run --bin bifurcata -- models                   # list the built-in models
cargo run --bin bifurcata -- compare out.json baseline.json --locations-only
```

`run` starts from the model's steady state (or `--start v1,v2,...`) and
continues in both directions (`--direction +`, `-` or `both`). A model with a
`[bifurcation]` block needs no `--param` or `--range`. The bifurcations found
are summarised on stderr; the run itself goes to `-o` (or stdout) as JSON, or
as CSV with `--format csv`. Other options: `--settings file.json`,
`--max-points n`, `--no-detect`, `--no-spectrum`, `--quiet`.

Exit codes: 0 finished, 1 stopped at a parameter or state bound (normal),
2 continuation failed, 3 no starting point, 4 model error, 5 `compare` found a
mismatch.

### An Antimony model, from Rust

```toml
[dependencies]
bifurcata = { git = "https://github.com/sys-bio/bifurcata-rs", features = ["antimony"] }
```

```rust
use bifurcata::antimony::AntimonyProblem;
use bifurcata::continuation::ContinuationOptions;
use bifurcata::run::EquilibriumRun;

let problem = AntimonyProblem::parse(model_text)?;
let lambda0 = problem.parameter_values();
let run = EquilibriumRun {
    problem: &problem,
    model_name: "tyson".into(),
    model_source: "tyson.ant".into(),
    u0: problem.find_steady_state(&lambda0)?,
    active: problem.parameter_index("m").unwrap(),
    lambda0,
    direction: 0, // both
    options: ContinuationOptions { parameter_min: 0.0, parameter_max: 2.0, ..Default::default() },
}
.run()?;
for branch in &run.branches {
    for b in &branch.bifurcations {
        println!("{} at m = {}", b.kind.abbreviation(), b.point.lambda[run.meta.active_parameter_indices[0]]);
    }
}
```

### Your own equations, step by step

Implement `BifurcationProblem` (the parameter derivative defaults to finite
differences), then drive the engine yourself — one point per `step()`:

```rust
use bifurcata::continuation::{ContinuationEngine, ContinuationOptions};
use bifurcata::equilibrium::EquilibriumCurve;
use bifurcata::matrix::Matrix;
use bifurcata::problem::BifurcationProblem;
use bifurcata::types::StepResult;

/// du/dt = λ + u − u³: an S-shaped curve with two folds.
struct Cubic;

impl BifurcationProblem for Cubic {
    fn state_dim(&self) -> usize { 1 }
    fn parameter_count(&self) -> usize { 1 }
    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
        r[0] = lambda[0] + u[0] - u[0].powi(3);
    }
    fn jacobian(&self, u: &[f64], _lambda: &[f64], j: &mut Matrix) {
        j.resize_zeroed(1, 1);
        j[(0, 0)] = 1.0 - 3.0 * u[0] * u[0];
    }
}

let options = ContinuationOptions { parameter_min: -1.0, parameter_max: 1.0, ..Default::default() };
let curve = EquilibriumCurve::new(Cubic, &[-1.0], 0, &options)?;
let x0 = curve.pack(&[-1.3247179572447460], -1.0); // the equilibrium at λ = −1
let mut engine = ContinuationEngine::new(curve, options);
engine.initialise(&x0, 1)?; // +1: increasing λ
while engine.step() == StepResult::Ok {
    // engine.branch().last_point() is the newest point: draw it here
}
for b in &engine.branch().bifurcations {
    println!("{} at λ = {:.6}", b.kind.abbreviation(), b.point.lambda[0]); // the folds at ±2/(3√3)
}
```

### Options

`ContinuationOptions` (defaults in brackets): `initial_step` [0.01],
`min_step` [1e-6], `max_step` [0.1], `parameter_min` / `parameter_max`,
`max_points` [1000], `detect_bifurcations` [true], `tol_residual` and
`tol_step` [1e-10], `max_newton_iterations` [10],
`target_newton_iterations` [4], `max_tangent_angle_deg` [30],
`predictor` [tangent] and `corrector` [PALC]. A settings file is the same
record as JSON, as written in a run's `run.options`.

## Building and testing

```
cargo test                                    # the core library
cargo test --features antimony                # plus Antimony models (fetches websim)
cargo check --target wasm32-unknown-unknown   # the browser build
cargo clippy --all-targets -- -D warnings
```

The baseline tests read the Delphi version's results in place from the Delphi
project (`BIFURCATA_DELPHI_DIR` points elsewhere) and skip when it is absent,
as on CI; the closed-form tests always run.

## Layout

| Module | What it holds |
|---|---|
| `problem` | The `BifurcationProblem` trait: what the library needs from a model |
| `continuation` | Options, the `DefiningSystem` trait, the PALC system and the engine |
| `equilibrium` | The equilibrium curve: stability and the fold, branch-point and Hopf test functions |
| `newton` | Damped Newton with line search and Levenberg–Marquardt fallback |
| `bialternate` | The bialternate product behind the Hopf test function |
| `linalg`, `matrix`, `bordered`, `complex` | Dense linear algebra over [faer](https://github.com/sarah-quinones/faer-rs) |
| `serialise` | `bifurcata/1` JSON, settings files, CSV, the comparator |
| `run`, `runspec`, `models` | A two-direction run, the `[bifurcation]` block, the built-in models |
| `antimony` | Antimony models through websim's model layer (feature `antimony`) |
| `test_problems` | Problems with known answers, used by the tests |

## Status

The plan and its milestone table are in websim's
[`docs/bifurcation-rust-plan.md`](https://github.com/sys-bio/websim/blob/main/docs/bifurcation-rust-plan.md).

| Milestone | Content | Status |
|---|---|---|
| M1 | Types, complex arithmetic, dense linear algebra, bordered solver, problem trait, test problems | Done |
| M2 | Newton, pseudo-arclength continuation, equilibria, fold/branch-point/Hopf detection, JSON output, Antimony models | Done |
| M3 | Normal forms, branch switching | Next |
| M4 | Codimension-2 fold and Hopf curves | — |
