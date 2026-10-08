# bifurcata-rs

A pure-Rust rewrite of the Delphi bifurcation library **Bifurcata**, crate
name `bifurcata`. https://github.com/sys-bio/bifurcata-rs

## Sources of truth

- **The plan and its progress**: `docs/bifurcation-rust-plan.md` in websim
  (`D:\Documents\rust\GUITest`, https://github.com/sys-bio/websim). Its
  milestone table (§8) is the status record; update it when a milestone ends.
- **The specification**: `BifurcationSpec.md` in the Delphi project at
  `D:\Documents\Embarcadero\Studio\Projects\Bifurcation_Delphi`. It is the
  authority on what the library does. Follow its intent, not its Delphi shapes.
- **The Delphi `CLAUDE.md`** in the same folder: numerical lessons and
  conventions to carry over deliberately. Read the relevant part before
  writing each module.
- **The Delphi source** (`src\`) and **tests** (`tests\`): read for detail and
  expected values; port tests check by check.
- **The Delphi project is read-only.** Never modify it.

## Layout

- `src/types.rs` — status codes, bifurcation kinds, curve points, branches.
  The `as_str` names are the `bifurcata/1` wire format: do not change them.
- `src/complex.rs` — complex numbers; division by Smith's algorithm.
- `src/matrix.rs` — dense matrix wrapping `faer::Mat` (column-major).
- `src/linalg.rs` — our own LU (completes on singular matrices; signed log
  determinant; transposed solve; Hager condition estimate), eigenvalues and
  left/right eigenvectors via faer, SVD and null vectors.
- `src/bordered.rs` — the bordered solver (`DirectBordered`).
- `src/problem.rs` — the `BifurcationProblem` trait and finite-difference helpers.
- `src/test_problems.rs` — hand-written problems with known answers.
- `src/newton.rs` — the one Newton (§2.1): Armijo line search, relative
  Levenberg–Marquardt fallback, both scalings (`typical_f_from_residual`,
  Delphi's; `scale_from_jacobian`, websim's); `EquilibriumSystem` (fixed λ).
- `src/continuation.rs` — `ContinuationOptions`, the `DefiningSystem` trait,
  `PalcSystem`, and `ContinuationEngine` (an iterator: `step()`), with both
  correctors, refinement and bisection, and the point/attempt/declined callbacks.
- `src/equilibrium.rs` — `EquilibriumCurve`: the equilibrium defining system,
  spectrum per point, ψ_LP / ψ_BP / ψ_H and their classification.
- `src/bialternate.rs` — `2A ⊙ I`, the normalised Hopf test function, Hopf vs
  neutral-saddle spectra.
- `src/serialise.rs` — `bifurcata/1` JSON (write, load), settings files, CSV,
  the comparator.
- `src/run.rs` — an equilibrium run as the harness does it (both directions,
  exit codes); `src/models.rs` — the built-in models;
  `src/bin/bifurcata.rs` — the console harness (`run`, `compare`, `models`).
- `src/antimony.rs` (feature `antimony`) — `AntimonyProblem` over websim's
  model crate (git dependency): reduced state, conserved totals as `_CSUM0…`.
- `tests/` — one file per ported Delphi suite (`stage0`, `newton`,
  `continuation`, `bialternate`, `scale`, `serialise`), `test_problems`,
  `baselines` (Delphi baselines, read in place; skip if absent), and the
  closed-form `pp2` and `lab2` (feature `antimony`).

## Conventions

- **Pure Rust, browser-compatible.** No C dependencies, no threads, no
  `std::time::Instant`. faer is used with `default-features = false` (its
  `rayon` feature does not compile for wasm), and
  `faer::set_global_parallelism(Par::Seq)` is set before faer routines. CI
  checks the wasm build.
- **faer is pre-1.0**: only `matrix.rs` and `linalg.rs` touch it.
- **Eigen conventions match LAPACK's `dgeev`**, so nothing downstream depends
  on the library: conjugate pairs adjacent, +Im first, the second member's
  eigenvectors exactly conjugate; each eigenvector unit-norm with its largest
  component (lowest index on a tie) real and positive. Eigenvalues are NOT
  sorted. Left eigenvectors satisfy `Aᵀ p = conj(μ) p`.
- **Errors** are `Result<_, SolveStatus>`; the codes and their wire names match
  the Delphi version (`LinAlgError` is `lapackError` on the wire).
- **Finite differences, not symbolic derivatives**, for rate-law Jacobians.
- **Testing discipline** (from the Delphi CLAUDE.md): after writing a test,
  break the code on purpose and watch it fail. Prefer independently known
  targets (closed forms, published values) over comparing two runs. Watch for
  vacuous checks.
- **When mutating, make cargo rebuild.** Restoring a file by `mv`-ing a backup
  over it gives it the backup's older mtime, so cargo keeps the mutant's build.
  Copy back and `touch`. And `cargo test` builds the `bifurcata` binary too:
  after a mutation run, `cargo build` before running `target/debug/bifurcata`.
  Both bit in M2 and produced convincing wrong numbers.
- **The source code is not rustfmt-formatted**; match the surrounding style.
- **Line endings are LF.** Python on Windows writes CRLF in text mode (the M2
  commit needed a follow-up for it): write bytes, or check `git diff --stat`.
- **websim uses this crate** (git dependency, feature `antimony`) for its
  Bifurcation view, so a change here reaches the app once it is pushed.

## Numerical facts (M2)

- **serde_json needs `float_roundtrip`** for an exact JSON round trip; its
  default parser is not correctly rounded.
- **Compare states by name.** libRoadRunner orders the independent species
  its own way, websim in model order; `compare_runs` maps by `stateNames`.
- **Location trials check the tangent angle** (`correct_on_branch`), as each
  step does: near a branch point a bisection trial can converge onto the
  crossing branch at a small |ψ|. A deviation from Delphi; no baseline moved.
- **The fold confirmation's eigenvalue tolerance is absolute**
  (√`eigenvalue_zero_tol`), as in Delphi, so a model with rates ~1e12 has its
  folds declined. Known limitation; worth a relative tolerance some day.
- **Normal forms are M3**: every record has confidence `spectrum`, so compare
  against Delphi baselines with `compare_normal_forms: false`
  (`--locations-only`).
- **The `[bifurcation]` block in `.ant` files is not read yet**, nor is
  `--trace` implemented; the baseline tests take their options from the
  baseline's own `run.options`.

## Next session

**M3**: directional derivatives, normal forms (fold `a`, Hopf `l1` in both
normalisations, BP), branch switching (`run --switch`); exit criteria in the
plan's §8. Read the Delphi `Bifurcata.NormalForms`, `Bifurcata.Derivatives`,
`Bifurcata.BranchSwitch` and their tests; restore the M3 checks noted at the
top of `tests/continuation.rs`; `ant_pp2_switch` becomes comparable.
(`docs/m2-briefing.md` is the record of how M2 was planned.)

## Progress

**M1 done** (October 2026): Stage 0 ported (8 suites, all passing; mutation-
tested), test problems checked against finite differences and closed forms,
CI with tests, clippy and the wasm build.

**M2 done** (October 2026): Newton, PALC and Moore–Penrose, equilibria with
stability, LP/BP/H/neutral-saddle detection with refinement and bisection,
`bifurcata/1` serialisation, settings, CSV, comparator, harness, Antimony
adapter. Delphi suites Newton, Continuation, Bialternate, Scale and Serialise
(equilibrium parts) ported; every mutation tried is caught. All 24 equilibrium
baselines (4 built-in, 19 `ant_*`, `csum_edelstein`) match in kinds, counts
and locations to 6 s.f. and in every sampled point to 1e-6; PP2's closed forms
and Lab 2's branch point to 1e-8. **Next: M3.**
