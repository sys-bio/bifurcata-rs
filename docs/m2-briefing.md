# M2 briefing — start here

Written at the end of the session that did M0 and M1, so that a fresh session
can do M2 without that conversation. Read this, then `CLAUDE.md`, then the
plan (§8 of `D:\Documents\rust\GUITest\docs\bifurcation-rust-plan.md`).

## Goal

**M2: equilibrium continuation with fold, branch-point and Hopf detection,
serialised in the `bifurcata/1` schema and checked against the Delphi
baselines.** Specification stage 1, plus the output needed to compare.

Exit criteria (plan §8):

- The Delphi test suites below, ported, passing, mutation-tested.
- Equilibrium baselines agree: bifurcation **parameter values to 6
  significant figures**, kinds and counts identical (normal-form
  coefficients are M3).
- **PP2 closed forms** reproduced: BP at p1 = 0.6, neutral saddle (not Hopf)
  at 0.4, BP at 0.821904345374, LP at 0.832929322234, Hopf at
  0.671593847479 with ω = 0.604782221942. (The last three need the
  coexistence/fish-only branches: start points `--start` as in the Delphi
  `regen.bat` `:antstart` calls; switching itself is M3.)

## Read first (all read-only)

Delphi project: `D:\Documents\Embarcadero\Studio\Projects\Bifurcation_Delphi`

| What | Where |
|---|---|
| Spec: architecture, engine, options | `BifurcationSpec.md` §4 (lines ~409–498) |
| Spec: PALC, Moore–Penrose, line search | §5 (~500–554) |
| Spec: starting point, stability, test functions | §6.1–6.3 (~556–638) |
| Spec: harness and JSON schema | §10.1, §10.3 (~1317–1483) |
| Lessons | `CLAUDE.md` — "Numerical facts worth not rediscovering", "Testing discipline", "PP2 is the closed-form model" |
| Newton | `src\Bifurcata.Newton.pas` (572 lines) |
| Engine, `IDefiningSystem`, PALC system | `src\Bifurcata.Continuation.pas` (IDefiningSystem ~95, engine ~209) |
| Equilibrium defining system, test functions | `src\Bifurcata.Equilibrium.pas` |
| Bialternate product | `src\Bifurcata.Bialternate.pas` |
| JSON writer/reader, comparator | `src\Bifurcata.Serialise.pas` |
| Run settings, `[bifurcation]` block | `src\Bifurcata.RunSpec.pas` |
| Harness `run` / `compare` | `harness\bifurcata.dpr` |

## Tests to port (≈ 230 checks)

| Delphi unit | Checks | Suites |
|---|---|---|
| `Tests.Newton` | 38 | test-problem Jacobians vs FD; basic convergence; Armijo; scaling; singular Jacobian at a fold; input validation |
| `Tests.Continuation` | 72 | equilibrium defining system; Brusselator branch + Hopf; fold rounded by PALC; streaming callback + abort; step adaptation; transcritical BP; predictor/corrector variants; Moore–Penrose agrees with PALC |
| `Tests.Bialternate` | 28 | spectrum = pairwise sums; zeros and their classification (Hopf vs neutral saddle); normalisation against overflow |
| `Tests.Scale` | 19 | bialternate cost; eigenvectors at higher n; continuation at n = 6, 10, 20 (Goodwin) |
| `Tests.Serialise` | 57 | round trip; `--no-spectrum`; comparator; settings file; CSV — port the equilibrium parts |

Port check by check, as `tests/stage0.rs` did. Then **mutate the code and
watch each new test fail** (M1 did four mutations; see git history).

## Defaults (Delphi `TContinuationOptions.Defaults`)

initial step 0.01, min 1e-6, max 0.1; increase ×1.3, decrease ×0.5; target
Newton iterations **4** (not 3: finite-difference Jacobians lose Newton's
quadratic rate); max Newton iterations 10; tolerances 1e-10 (residual and
step); line search on; max points 1000; parameter bounds ±1e300; state bound
1e12; detect bifurcations on; bisection tolerance 1e-8, max 40 bisections;
eigenvalue zero tolerance 1e-8; max tangent angle 30°; predictor tangent;
corrector PALC. (`AdaptMeshEvery`, `UseCondensation` are periodic-orbit
options, not M2.)

## Decisions

Settled (with the user): finite-difference Jacobians, never symbolic; our own
Newton, not a crate; eigen conventions as in `linalg.rs`; wire names fixed.

To decide at the start of M2 — recommendations:

1. **One Newton (spec §2.1).** websim already has a Rust port of
   `Bifurcata.Newton` with M0's fixes (`D:\Documents\rust\GUITest\crates\model\src\newton.rs`):
   Jacobian-based scaling, relative Levenberg–Marquardt damping, an rcond
   switch, and `NonlinearSystem::admissible`. **Recommend: move it into
   bifurcata as `src/newton.rs`** (over bifurcata's `Matrix` and `Lu`), keep
   both scalings (Delphi's `typical_f_from_residual`, which `Tests.Newton`
   expects, and `scale_from_jacobian`), and later have websim's steady-state
   solver use bifurcata's. Check `Tests.Newton`'s expectations against the M0
   changes; where they conflict, the M0 lessons are evidence (see websim
   `CLAUDE.md`, "Numerical facts").
2. **Engine shape.** `trait DefiningSystem` (spec §4.1: `system_dim`,
   `residual`, `jacobian` (N−1)×N, `test_function_count`, `test_functions(x,
   tangent)`, `test_function_kind`, `process_singularity`, `decode`) and
   `ContinuationEngine::step() -> StepResult`, an iterator so the GUI can draw
   as it grows and stop at any time (essential in the browser).
3. **Hopf test**: bialternate product with `linalg::normalised_determinant`
   (sign × geometric mean; already in M1). A zero with a real pair ±σ is a
   **neutral saddle** — recorded, not called Hopf.
4. **Location**: bisection on the bracket, preceded by Delphi's
   `RefineAndLocate` (re-walk the bracket in real continuation steps first —
   took Lab 2's branch point from 1.2% to nine figures).
5. **JSON**: `serde` + `serde_json`, schema `bifurcata/1`. Field names from a
   real baseline (`baselines\ant_brusselator.json`): top level `schema, model
   {name, source, stateNames, independentCount}, run {activeParameters, range,
   direction, options{…}}, branches [{id, parentId, originBifurcationId,
   points [{s, u, lambda, unstableDim, stepSize, newtonIterations}],
   terminationReason}], bifurcations [{id, branchId, kind, s, u, lambda,
   normalForm{…}, classification, confidence, detail}], status`. Delphi files
   start with a UTF-8 BOM — strip `'\u{feff}'` when reading; write without one.
6. **Antimony adapter** (feature `antimony`): an optional dependency on
   websim's model crate — `websim-model = { git =
   "https://github.com/sys-bio/websim", optional = true }` (cargo finds the
   package inside the workspace), with a local `[patch]` or path during
   development. Map: state = the reduced (independent) species; residual =
   `Model::reduced_rates`; Jacobian = `Model::reduced_jacobian`; start =
   `steady::find_steady_state`. **Gap to fill in websim**: a public list of
   parameter names and indices (`parameter_values()` indexes all variables,
   rules included; `parameter_index(name)` exists) and conserved totals as
   extra parameters named `_CSUM0, _CSUM1, …` (needed by `csum_edelstein`).

## Baselines for M2

`baselines\*.json`, read in place. The `delphi_models.rs` test in websim shows
the pattern (env var `BIFURCATA_DELPHI_DIR`, skip when absent, strip BOM).

- Built-in problems (`run <name>`): `brusselator`, `selkov`, `saddlenode`,
  `transcritical`.
- Antimony models (`run <file> --param P --range a:b`): the 20 `ant_*.json`;
  each file's `run` section has the parameter, range and options. pp2's three
  use explicit starts (`0,0` and `0.33333333333333,2`).
- Conserved totals: `moiety3`, `csum_edelstein` (parameter `_CSUM0`).

Compare **bifurcations first** (kind, count, location to 6 s.f.); points with
a looser tolerance — finite-difference details differ from libRoadRunner's,
which can change the step sequence. websim M0 already reproduces every
`ant_*` starting state to 1.3e-11, so starting points are not the risk.

## Suggested order

1. Newton (decision 1) + `Tests.Newton`.
2. `DefiningSystem`, the PALC system, the engine; `TFoldProblem` round its
   fold, Brusselator branch — `Tests.Continuation` parts without detection.
3. Stability (eigenvalues per point, unstable dimension).
4. Test functions (LP = tangent's parameter component; BP = bordered
   determinant; H = bialternate) + location; `Tests.Bialternate`, the rest of
   `Tests.Continuation`, `Tests.Scale`.
5. Serialisation + comparator; `Tests.Serialise` (equilibrium parts).
6. The `antimony` adapter (+ the websim gap), a `bifurcata` binary with `run`
   and `compare`, and a baseline test like websim's `delphi_models.rs`.
7. PP2 closed forms. Update the plan's §8 table, `CLAUDE.md` and README.

## Pitfalls already found

- Browser: no threads, no `std::time::Instant` (use `web-time` or an injected
  clock for time budgets), faer without default features.
- Eigenvalues are **not sorted**; pick by property, not position (a test was
  wrong about this in M1).
- Ties: "largest component" conventions need a tie rule (lowest index).
- Finite-difference steps must not be relative to zero, and residual scaling
  by |F(x0)| breaks when a rate starts at zero (websim `CLAUDE.md`).
- Edit files with non-ASCII text using the Edit tool, not shell `sed`.
