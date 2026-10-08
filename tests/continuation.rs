//! The continuation engine and equilibrium detection (specification §5.1,
//! §6.1–6.3), ported check by check from the Delphi
//! `Bifurcata.Tests.Continuation`.
//!
//! The load-bearing cases: the Brusselator, with a Hopf at B = 1 + A² and
//! ω = A and the branch u = (A, B/A), all exact; and λ = u², which turns back
//! on itself at λ = 0 — natural-parameter continuation cannot pass it, PALC
//! must round it. That is the test that justifies pseudo-arclength.
//!
//! Deferred to M3 (normal forms): the checks that the Brusselator's Hopf
//! carries l1 = −1/2, is classified supercritical with confidence
//! `coefficient`, and that the fold carries a = −1.

use std::cell::RefCell;
use std::rc::Rc;

use bifurcata::continuation::{ContinuationEngine, ContinuationOptions, DefiningSystem};
use bifurcata::equilibrium::{EquilibriumCurve, TF_BRANCH_POINT, TF_FOLD, TF_HOPF};
use bifurcata::matrix::{Matrix, norm_inf};
use bifurcata::test_problems::*;
use bifurcata::types::*;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!((actual - expected).abs() <= tol, "{what}: got {actual}, expected {expected} (tolerance {tol:e})");
}

fn count(branch: &Branch, kind: BifurcationKind) -> usize {
    branch.bifurcations.iter().filter(|b| b.kind == kind).count()
}

fn first(branch: &Branch, kind: BifurcationKind) -> Option<&BifurcationInfo> {
    branch.bifurcations.iter().find(|b| b.kind == kind)
}

/// Step until the branch ends or `max_steps` is reached.
fn run_to_end<S: DefiningSystem>(engine: &mut ContinuationEngine<S>, max_steps: usize) {
    for _ in 0..max_steps {
        if engine.step() != StepResult::Ok {
            break;
        }
    }
}

fn brusselator_engine(options: ContinuationOptions) -> ContinuationEngine<EquilibriumCurve<Brusselator>> {
    let curve = EquilibriumCurve::new(Brusselator, &[1.0, 1.2], 1, &options).unwrap();
    let x0 = curve.pack(&Brusselator::equilibrium(1.0, 1.2), 1.2);
    let mut engine = ContinuationEngine::new(curve, options);
    engine.initialise(&x0, 1).expect("the branch initialises at B = 1.2");
    engine
}

#[test]
fn equilibrium_defining_system() {
    let curve = EquilibriumCurve::new(Brusselator, &[1.0, 2.0], 1, &ContinuationOptions::default()).unwrap();
    assert_eq!(curve.system_dim(), 3, "N = n + 1 = 3 for the Brusselator");
    assert_eq!(curve.active_parameter_index(), 2, "the active parameter is the last component of x");

    let x = curve.pack(&Brusselator::equilibrium(1.0, 2.0), 2.0);
    let mut g = [0.0; 2];
    curve.residual(&x, &mut g);
    close(norm_inf(&g), 0.0, 1e-14, "the residual vanishes at the analytic equilibrium");

    // DG is n × (n+1) and matches finite differences of G, including the
    // parameter column — the easiest one to get wrong.
    let x = curve.pack(&[1.3, 2.4], 2.7);
    let mut dg = Matrix::zeros(0, 0);
    curve.jacobian(&x, &mut dg);
    assert_eq!((dg.rows(), dg.cols()), (2, 3), "DG is n × (n+1)");
    let mut worst = 0.0f64;
    for col in 0..3 {
        let h = 1e-6 * x[col].abs().max(1.0);
        let (mut xp, mut xm) = (x.clone(), x.clone());
        xp[col] += h;
        xm[col] -= h;
        let h = (xp[col] - xm[col]) / 2.0;
        let (mut gp, mut gm) = ([0.0; 2], [0.0; 2]);
        curve.residual(&xp, &mut gp);
        curve.residual(&xm, &mut gm);
        for row in 0..2 {
            worst = worst.max((dg[(row, col)] - (gp[row] - gm[row]) / (2.0 * h)).abs());
        }
    }
    close(worst, 0.0, 1e-5, "DG matches finite differences of G, the parameter column included");

    // ψ_H vanishes at the exact Hopf point and not away from it.
    let x = curve.pack(&Brusselator::equilibrium(1.0, 2.0), 2.0);
    let tau = [0.0, 1.0, 0.0];
    let mut psi = [0.0; 3];
    curve.test_functions(&x, &tau, &mut psi);
    close(psi[TF_HOPF], 0.0, 1e-10, "ψ_H vanishes at the analytic Hopf point B = 1 + A²");
    let x = curve.pack(&Brusselator::equilibrium(1.0, 1.5), 1.5);
    curve.test_functions(&x, &tau, &mut psi);
    assert!(psi[TF_HOPF].abs() > 1e-6, "ψ_H is nonzero away from the Hopf point");
}

#[test]
fn brusselator_branch_and_hopf() {
    let options = ContinuationOptions { initial_step: 0.02, max_step: 0.05, max_points: 400, parameter_min: 1.0, parameter_max: 4.0, ..Default::default() };
    let mut engine = brusselator_engine(options);
    run_to_end(&mut engine, 500);
    let branch = engine.branch();

    assert!(branch.points.len() > 20, "the branch has a reasonable number of points ({})", branch.points.len());
    assert_eq!(branch.termination, TerminationReason::ParameterBound, "the branch stopped at the parameter bound");

    // Every point on the analytic branch u = (A, B/A), A = 1.
    let worst_x = branch.points.iter().map(|p| (p.u[0] - 1.0).abs()).fold(0.0, f64::max);
    let worst_y = branch.points.iter().map(|p| (p.u[1] - p.lambda[1]).abs()).fold(0.0, f64::max);
    close(worst_x, 0.0, 1e-8, "every point has X = A exactly");
    close(worst_y, 0.0, 1e-8, "every point has Y = B/A exactly");

    // The Hopf at B = 1 + A² = 2, ω = A = 1; §11.2 asks for six figures.
    let hopf = first(branch, BifurcationKind::Hopf).expect("a Hopf point was detected");
    close(hopf.point.lambda[1], 2.0, 1e-6, "Hopf located at B = 2 to six significant figures");
    close(hopf.normal_form.get_or("omega", 0.0), 1.0, 1e-5, "Hopf frequency ω = 1");
    close(hopf.point.u[0], 1.0, 1e-6, "Hopf point X = 1");
    close(hopf.point.u[1], 2.0, 1e-6, "Hopf point Y = 2");

    assert_eq!(count(branch, BifurcationKind::Hopf), 1, "exactly one Hopf point on the branch");
    assert_eq!(count(branch, BifurcationKind::Fold), 0, "no fold reported (the Brusselator determinant is A² > 0)");

    // Stability changes across it.
    assert_eq!(branch.points[0].unstable_dim, 0, "the branch starts stable at B = 1.2");
    assert_eq!(branch.last_point().unwrap().unstable_dim, 2, "the branch ends unstable with a complex pair crossed");
}

fn fold_options() -> ContinuationOptions {
    ContinuationOptions { initial_step: 0.05, max_step: 0.1, max_points: 400, parameter_min: -0.5, parameter_max: 4.0, ..Default::default() }
}

/// Runs the fold problem from the upper arm towards the fold.
fn fold_branch(options: ContinuationOptions) -> Branch {
    let curve = EquilibriumCurve::new(FoldProblem, &[2.0], 0, &options).unwrap();
    let x0 = curve.pack(&[2f64.sqrt()], 2.0);
    let mut engine = ContinuationEngine::new(curve, options);
    engine.initialise(&x0, -1).expect("the branch initialises on the upper arm");
    run_to_end(&mut engine, 500);
    engine.into_branch()
}

#[test]
fn pseudo_arclength_rounds_the_fold() {
    let branch = fold_branch(fold_options());

    let worst = branch.points.iter().map(|p| (p.lambda[0] - p.u[0] * p.u[0]).abs()).fold(0.0, f64::max);
    close(worst, 0.0, 1e-8, "every point satisfies λ = u²");

    // The decisive assertion: both arms visited, so it went round the fold.
    let saw_positive = branch.points.iter().any(|p| p.u[0] > 0.1);
    let saw_negative = branch.points.iter().any(|p| p.u[0] < -0.1);
    assert!(saw_positive && saw_negative, "the branch traversed both arms of the fold");
    let max_u = branch.points.iter().map(|p| p.u[0]).fold(f64::MIN, f64::max);
    let min_u = branch.points.iter().map(|p| p.u[0]).fold(f64::MAX, f64::min);
    assert!(max_u > 1.0, "reached well up the upper arm");
    assert!(min_u < -1.0, "reached well down the lower arm");
    // Points are spaced by arclength, so none need sit at the fold itself.
    let min_lambda = branch.points.iter().map(|p| p.lambda[0]).fold(f64::MAX, f64::min);
    assert!(min_lambda < 0.01, "the branch approached the fold closely (nearest point at λ = {min_lambda:.3e})");

    let fold = first(&branch, BifurcationKind::Fold).expect("the fold was detected");
    close(fold.point.lambda[0], 0.0, 1e-6, "fold located at λ = 0");
    close(fold.point.u[0], 0.0, 1e-4, "fold located at u = 0");

    assert_eq!(branch.termination, TerminationReason::ParameterBound, "the branch ended at a parameter bound");
}

#[test]
fn transcritical_branch_point() {
    let options = ContinuationOptions { initial_step: 0.02, max_step: 0.05, max_points: 400, parameter_min: -1.5, parameter_max: 1.5, ..Default::default() };
    let curve = EquilibriumCurve::new(TranscriticalProblem, &[-1.0], 0, &options).unwrap();

    // The test functions discriminate before any continuation: on u = 0, ψ_BP
    // is proportional to λ and ψ_LP is not.
    let tau = [0.0, 1.0];
    let mut psi = [0.0; 3];
    curve.test_functions(&curve.pack(&[0.0], -1.0), &tau, &mut psi);
    assert!(psi[TF_BRANCH_POINT].abs() > 1e-6, "ψ_BP is nonzero away from the branch point");
    assert!(psi[TF_FOLD].abs() > 0.5, "ψ_LP stays at unity on u = 0 (this is not a fold)");
    curve.test_functions(&curve.pack(&[0.0], 0.0), &tau, &mut psi);
    close(psi[TF_BRANCH_POINT], 0.0, 1e-10, "ψ_BP vanishes at the branch point λ = 0");

    let x0 = curve.pack(&[0.0], -1.0);
    let mut engine = ContinuationEngine::new(curve, options);
    engine.initialise(&x0, 1).expect("the branch initialises on u = 0 at λ = −1");
    run_to_end(&mut engine, 500);
    let branch = engine.branch();

    let worst = branch.points.iter().map(|p| p.u[0].abs()).fold(0.0, f64::max);
    close(worst, 0.0, 1e-7, "the branch stayed on u = 0");

    let bp = first(branch, BifurcationKind::BranchPoint).expect("the branch point was detected");
    close(bp.point.lambda[0], 0.0, 1e-6, "branch point located at λ = 0");
    close(bp.point.u[0], 0.0, 1e-6, "branch point located at u = 0");
    assert_eq!(bp.point.tangent.len(), 2, "the record carries the tangent it arrived on (for branch switching)");

    // Not misreported as a fold: τ_λ never vanishes on this branch.
    assert_eq!(count(branch, BifurcationKind::Fold), 0, "no fold reported at a transcritical point");
    assert_eq!(count(branch, BifurcationKind::BranchPoint), 1, "exactly one branch point on the branch");

    // A real eigenvalue crossing: the unstable dimension changes by one.
    assert_eq!(branch.points[0].unstable_dim, 0, "u = 0 is stable for λ < 0");
    assert_eq!(branch.last_point().unwrap().unstable_dim, 1, "u = 0 is unstable for λ > 0");
}

#[test]
fn neutral_saddle_is_recorded_not_called_hopf() {
    // On u = 0 the spectrum is {λ, −1}: a real pair summing to zero at λ = 1.
    let options = ContinuationOptions { initial_step: 0.02, max_step: 0.05, parameter_min: 0.5, parameter_max: 1.5, ..Default::default() };
    let curve = EquilibriumCurve::new(NeutralSaddleProblem, &[0.5], 0, &options).unwrap();
    let x0 = curve.pack(&[0.0, 0.0], 0.5);
    let mut engine = ContinuationEngine::new(curve, options);
    engine.initialise(&x0, 1).unwrap();
    run_to_end(&mut engine, 500);
    let branch = engine.branch();

    assert_eq!(count(branch, BifurcationKind::Hopf), 0, "a real pair summing to zero is not a Hopf");
    assert_eq!(branch.bifurcations.len(), 1, "exactly one record on the branch: {:?}", branch.bifurcations.iter().map(|b| b.kind).collect::<Vec<_>>());
    let ns = first(branch, BifurcationKind::NeutralSaddle).expect("the zero of ψ_H is recorded as a neutral saddle");
    close(ns.point.lambda[0], 1.0, 1e-6, "neutral saddle located at λ = 1");
    assert_eq!(ns.confidence, Confidence::Spectrum, "classified from a recognised spectrum, not left unclassified");
    // Nothing changes stability there: one unstable direction throughout.
    assert!(branch.points.iter().all(|p| p.unstable_dim == 1), "the unstable dimension stays 1 across a neutral saddle");
}

/// §6.1: the corrector is scaled by the row norms of DG. `1e12 (λ − u²)` has
/// residuals whose rounding error alone (about 1e12 · ε) exceeds even the 1e-5
/// Newton accepts once its step has collapsed, so without the scaling the
/// corrector can never converge and the
/// branch dies at its first step.
#[test]
fn corrector_scaling_survives_large_rates() {
    struct BigFold;
    impl bifurcata::problem::BifurcationProblem for BigFold {
        fn state_dim(&self) -> usize {
            1
        }
        fn parameter_count(&self) -> usize {
            1
        }
        fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
            r[0] = 1e12 * (lambda[0] - u[0] * u[0]);
        }
        fn jacobian(&self, u: &[f64], _lambda: &[f64], j: &mut Matrix) {
            j.resize_zeroed(1, 1);
            j[(0, 0)] = -2e12 * u[0];
        }
    }
    let options = fold_options();
    let curve = EquilibriumCurve::new(BigFold, &[2.0], 0, &options).unwrap();
    let x0 = curve.pack(&[2f64.sqrt()], 2.0);
    let mut engine = ContinuationEngine::new(curve, options);
    engine.initialise(&x0, -1).unwrap();
    run_to_end(&mut engine, 500);
    let branch = engine.branch();
    assert_eq!(branch.termination, TerminationReason::ParameterBound, "the branch ran to its bound: {}", branch.termination_detail);
    assert!(branch.points.iter().any(|p| p.u[0] < -1.0), "and went round the fold onto the lower arm");
    let worst = branch.points.iter().map(|p| (p.lambda[0] - p.u[0] * p.u[0]).abs()).fold(0.0, f64::max);
    close(worst, 0.0, 1e-8, "every point satisfies λ = u²");
    // Not asserted: the fold record. Its confirmation compares the smallest
    // |eigenvalue| with an absolute √eigenvalue_zero_tol, as the Delphi version
    // does, so at this scale it is declined — a known limitation.
}

/// Whether the fold problem's branch rounds the fold, its worst residual, and its point count.
fn round_fold_with(corrector: CorrectorKind, predictor: PredictorKind) -> (bool, f64, usize) {
    let branch = fold_branch(ContinuationOptions { corrector, predictor, ..fold_options() });
    let worst = branch.points.iter().map(|p| (p.lambda[0] - p.u[0] * p.u[0]).abs()).fold(0.0, f64::max);
    let rounded = branch.points.iter().any(|p| p.u[0] > 0.1) && branch.points.iter().any(|p| p.u[0] < -0.1);
    (rounded, worst, branch.points.len())
}

#[test]
fn corrector_and_predictor_variants() {
    let (ok, worst, palc_points) = round_fold_with(CorrectorKind::Palc, PredictorKind::Tangent);
    assert!(ok, "PALC + tangent rounds the fold");
    close(worst, 0.0, 1e-8, "PALC + tangent stays on λ = u²");

    // Moore–Penrose corrects to the nearest point, so only the geometry must agree.
    let (ok, worst, mp_points) = round_fold_with(CorrectorKind::MoorePenrose, PredictorKind::Tangent);
    assert!(ok, "Moore–Penrose rounds the fold");
    close(worst, 0.0, 1e-8, "Moore–Penrose stays on λ = u²");

    let (ok, worst, secant_points) = round_fold_with(CorrectorKind::Palc, PredictorKind::Secant);
    assert!(ok, "the secant predictor rounds the fold");
    close(worst, 0.0, 1e-8, "the secant predictor stays on λ = u²");

    assert!(palc_points > 10 && mp_points > 10 && secant_points > 10, "all three produced real branches ({palc_points}, {mp_points}, {secant_points} points)");
}

/// Equal point counts are also what a silent fall-through to PALC would look
/// like, so check the requested code path actually ran.
#[test]
fn variant_selection_takes_effect() {
    let base = ContinuationOptions { max_points: 20, parameter_max: 4.0, detect_bifurcations: false, ..Default::default() };

    let mut engine = brusselator_engine(base.clone());
    engine.step();
    assert_eq!(engine.last_corrector_used(), CorrectorKind::Palc, "PALC ran the PALC corrector");
    assert_eq!(engine.last_predictor_used(), PredictorKind::Tangent, "Tangent ran the tangent predictor");

    let mut engine = brusselator_engine(ContinuationOptions { corrector: CorrectorKind::MoorePenrose, ..base.clone() });
    engine.step();
    assert_eq!(engine.last_corrector_used(), CorrectorKind::MoorePenrose, "MoorePenrose ran the Moore–Penrose corrector, not PALC");

    let mut engine = brusselator_engine(ContinuationOptions { predictor: PredictorKind::Secant, ..base });
    engine.step();
    assert_eq!(engine.last_predictor_used(), PredictorKind::Tangent, "the first step falls back to the tangent, having no secant yet");
    engine.step();
    assert_eq!(engine.last_predictor_used(), PredictorKind::Secant, "the second step uses the secant predictor");
}

#[test]
fn moore_penrose_agrees_with_palc_on_the_hopf() {
    let options = ContinuationOptions {
        initial_step: 0.02,
        max_step: 0.05,
        max_points: 400,
        parameter_min: 1.0,
        parameter_max: 4.0,
        corrector: CorrectorKind::MoorePenrose,
        ..Default::default()
    };
    let mut engine = brusselator_engine(options);
    run_to_end(&mut engine, 500);
    // §11.3: the same bifurcation found two ways must agree.
    let hopf = first(engine.branch(), BifurcationKind::Hopf).expect("Moore–Penrose found the Hopf point");
    close(hopf.point.lambda[1], 2.0, 1e-6, "Moore–Penrose locates the Hopf at B = 2, as PALC does");
    close(hopf.normal_form.get_or("omega", 0.0), 1.0, 1e-5, "Moore–Penrose agrees on ω = 1");
}

#[test]
fn streaming_callback_and_user_abort() {
    let options = ContinuationOptions { max_points: 200, parameter_max: 4.0, detect_bifurcations: false, ..Default::default() };
    let curve = EquilibriumCurve::new(Brusselator, &[1.0, 1.2], 1, &options).unwrap();
    let x0 = curve.pack(&Brusselator::equilibrium(1.0, 1.2), 1.2);
    let mut engine = ContinuationEngine::new(curve, options);

    let seen = Rc::new(RefCell::new(0usize));
    let counter = Rc::clone(&seen);
    engine.on_point(move |_| {
        *counter.borrow_mut() += 1;
        *counter.borrow() >= 12
    });
    engine.initialise(&x0, 1).unwrap();
    run_to_end(&mut engine, 300);

    assert_eq!(*seen.borrow(), 12, "the callback fired for every point");
    assert_eq!(engine.branch().points.len(), 12, "the branch holds exactly the streamed points");
    assert_eq!(engine.branch().termination, TerminationReason::UserAborted, "the abort was recorded as the termination reason");
}

#[test]
fn step_size_adaptation() {
    let options = ContinuationOptions { initial_step: 0.005, max_step: 0.2, max_points: 60, parameter_max: 10.0, detect_bifurcations: false, ..Default::default() };
    let max_step = options.max_step;
    let mut engine = brusselator_engine(options);
    run_to_end(&mut engine, 100);
    let points = &engine.branch().points;

    // On an easy branch the corrector meets the target, so the step grows.
    let grew = points.windows(2).skip(1).any(|w| w[0].step_size > 0.0 && w[1].step_size > w[0].step_size * 1.01);
    assert!(grew, "the step size grew on an easy branch");
    assert!(points.iter().all(|p| p.step_size <= max_step + 1e-12), "the step size never exceeded max_step");
    assert_eq!(engine.branch().termination, TerminationReason::MaxPoints, "the branch stopped at its point budget");
}
