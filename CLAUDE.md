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
- `tests/stage0.rs` — the Delphi Stage 0 suite, ported; `tests/test_problems.rs`.

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

## Next session

**Start with `docs/m2-briefing.md`**: the goal, reading list, tests to port,
defaults, decisions (settled and to make), baselines and an ordered plan for M2.

## Progress

**M1 done** (October 2026): Stage 0 ported (8 suites, all passing; mutation-
tested), test problems checked against finite differences and closed forms,
CI with tests, clippy and the wasm build. **Next: M2** — Newton, the PALC
continuation engine, the equilibrium defining system, stability, fold /
branch-point / Hopf detection, serialisation and the comparator, and the
`antimony` adapter to websim's model crate.
