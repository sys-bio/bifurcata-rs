//! Newton (specification §5.3, §6.1), ported check by check from the Delphi
//! `Bifurcata.Tests.Newton`. Assertions are against analytic values: the
//! Brusselator equilibrium is (A, B/A) exactly, the cubic's roots are known,
//! and the badly scaled problem has an exact solution by construction.
//!
//! Not ported: "a nil system is rejected" (no nil in Rust). The test-problem
//! Jacobian checks of the Delphi suite's first section are in
//! `tests/test_problems.rs`.

use bifurcata::matrix::norm_inf;
use bifurcata::newton::{EquilibriumSystem, NewtonOptions, NewtonSolver};
use bifurcata::problem::BifurcationProblem;
use bifurcata::test_problems::*;
use bifurcata::types::SolveCode;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!((actual - expected).abs() <= tol, "{what}: got {actual}, expected {expected} (tolerance {tol:e})");
}

#[test]
fn residual_vanishes_at_the_analytic_brusselator_equilibrium() {
    let mut r = [0.0; 2];
    Brusselator.residual(&Brusselator::equilibrium(1.0, 2.0), &[1.0, 2.0], &mut r);
    close(norm_inf(&r), 0.0, 1e-15, "residual vanishes at (A, B/A)");
}

#[test]
fn basic_convergence() {
    let sys = EquilibriumSystem::new(&Brusselator, &[1.0, 2.0]);
    let newton = NewtonSolver::default();
    let expected = Brusselator::equilibrium(1.0, 2.0);

    let mut x = vec![0.7, 1.4];
    let report = newton.solve(&sys, &mut x).expect("converges from a nearby start");
    close(x[0], expected[0], 1e-9, "X converged to A");
    close(x[1], expected[1], 1e-9, "Y converged to B/A");
    assert!(report.iterations <= 8, "converged in few iterations (took {})", report.iterations);

    // An exact start costs zero iterations: the corrector calls this at every point.
    let mut x = expected.to_vec();
    let report = newton.solve(&sys, &mut x).expect("an exact starting point is accepted");
    assert_eq!(report.iterations, 0, "an exact starting point costs zero iterations");

    // A far start: only that it arrives.
    let mut x = vec![5.0, 0.05];
    newton.solve(&sys, &mut x).expect("converges from a distant start");
    close(x[0], expected[0], 1e-8, "distant start: X correct");
    close(x[1], expected[1], 1e-8, "distant start: Y correct");
}

#[test]
fn armijo_line_search() {
    // arctan(u) = 0: undamped Newton diverges from beyond about 1.39. Both runs
    // get the same generous budget, so only the line search differs.
    let sys = EquilibriumSystem::new(&ArctanProblem, &[0.0]);
    let base = NewtonOptions { max_iterations: 50, use_levenberg_fallback: false, ..Default::default() };

    let mut x = vec![2.0];
    let with = NewtonSolver::new(NewtonOptions { use_line_search: true, ..base.clone() }).solve(&sys, &mut x);
    assert!(with.is_ok(), "the line search converges on arctan from u = 2: {with:?}");
    close(x[0], 0.0, 1e-8, "converged to the root u = 0");

    let mut x = vec![2.0];
    let without = NewtonSolver::new(NewtonOptions { use_line_search: false, ..base }).solve(&sys, &mut x);
    let failure = without.expect_err("undamped Newton diverges on arctan from the same start");
    // A failure says which condition fired (§9.1).
    assert!(!failure.detail.is_empty(), "the failure carries a diagnostic string");
    assert!(failure.detail.contains("Newton"), "the diagnostic names the solver: {}", failure.detail);

    // A slow but always-convergent case: the line search must not obstruct it.
    let sys = EquilibriumSystem::new(&CubicProblem, &[0.0]);
    let mut x = vec![12.0];
    NewtonSolver::new(NewtonOptions { max_iterations: 40, ..Default::default() })
        .solve(&sys, &mut x)
        .expect("the cubic converges from u = 12 given enough iterations");
    assert!((x[0].abs() - 1.0).abs() < 1e-8, "converged to a root at |u| = 1, got {}", x[0]);
}

#[test]
fn variable_and_residual_scaling() {
    // Residuals differ by 1e18: the second equation's is below the absolute
    // tolerance however wrong V is, so an unscaled test cannot see it.
    let problem = BadlyScaledProblem { scale_u: 1e6, scale_v: 1e-12, target_u: 2.0, target_v: 3.0 };
    let sys = EquilibriumSystem::new(&problem, &[0.0]);

    let mut newton = NewtonSolver::default();
    newton.typical_f_from_residual(&sys, &[0.0, 0.0], 1e-12);
    let mut x = vec![0.0, 0.0];
    newton.solve(&sys, &mut x).expect("the scaled solve converges on the 1e18-range problem");
    close(x[0], 2.0, 1e-9, "scaled solve: U correct");
    close(x[1], 3.0, 1e-9, "scaled solve: V correct (the equation an unscaled test would ignore)");

    // Unscaled, the solver reports success at once with V still wrong. Not a
    // bug: it is what "unscaled convergence tests are meaningless" means.
    let mut x = vec![2.0, 0.0];
    NewtonSolver::default().solve(&sys, &mut x).expect("the unscaled solve reports success");
    assert!((x[1] - 3.0).abs() > 1.0, "the unscaled solve leaves V wrong, which is why scaling is required");

    // The Jacobian scaling, websim's recommended route, solves it too.
    let mut newton = NewtonSolver::default();
    newton.scale_from_jacobian(&sys, &[1.0, 1.0]);
    let mut x = vec![1.0, 1.0];
    newton.solve(&sys, &mut x).expect("the Jacobian-scaled solve converges");
    close(x[1], 3.0, 1e-9, "Jacobian-scaled solve: V correct");
}

#[test]
fn singular_jacobian_at_a_fold() {
    // λ − u² at λ = 0: the only equilibrium is u = 0, where J = −2u is singular.
    let sys = EquilibriumSystem::new(&FoldProblem, &[0.0]);
    let without_lm = NewtonOptions { use_levenberg_fallback: false, ..Default::default() };

    // The residual is already zero, so this is converged, not a factorisation
    // failure: the early-out comes before the Jacobian is touched.
    let mut x = vec![0.0];
    let report = NewtonSolver::new(without_lm.clone()).solve(&sys, &mut x).expect("a zero residual at a singular point is converged");
    assert_eq!(report.iterations, 0, "and costs no iterations");

    // Approaching the fold the root is double: linear convergence, halving each step.
    let mut x = vec![1e-3];
    let report = NewtonSolver::new(NewtonOptions { max_iterations: 60, ..without_lm.clone() })
        .solve(&sys, &mut x)
        .expect("Newton reaches the fold from nearby (linear rate)");
    close(x[0], 0.0, 1e-4, "converged towards the fold point u = 0");
    assert!(report.iterations > 2, "a double root costs more than a quadratic one (took {})", report.iterations);

    // An EXACTLY singular Jacobian with a nonzero residual and a solution that
    // exists: the case the fallback is for.
    let sys = EquilibriumSystem::new(&SingularConsistentProblem, &[0.0]);
    let mut x = vec![0.0, 0.0];
    let failure = NewtonSolver::new(without_lm).solve(&sys, &mut x).expect_err("without the fallback, an exactly singular Jacobian fails");
    assert_eq!(failure.code, SolveCode::SingularJacobian, "and is classified as SingularJacobian");
    assert!(failure.detail.contains("Levenberg"), "the diagnostic points at the disabled fallback: {}", failure.detail);

    let mut x = vec![0.0, 0.0];
    let report = NewtonSolver::new(NewtonOptions { max_iterations: 60, ..Default::default() })
        .solve(&sys, &mut x)
        .expect("Levenberg–Marquardt solves what plain Newton could not factor");
    close(x[0] + x[1], 1.0, 1e-8, "landed on the solution line u + v = 1");
    assert!(report.used_levenberg, "and says it used Levenberg–Marquardt");
}

/// From websim's port: a Jacobian whose reciprocal condition number (about
/// 1e-14) is below the 1e-12 threshold. Levenberg–Marquardt takes over and
/// still gets the residual down.
#[test]
fn levenberg_marquardt_takes_over_when_ill_conditioned() {
    struct IllConditioned;
    impl BifurcationProblem for IllConditioned {
        fn state_dim(&self) -> usize {
            2
        }
        fn parameter_count(&self) -> usize {
            1
        }
        fn residual(&self, u: &[f64], _lambda: &[f64], r: &mut [f64]) {
            r[0] = u[0] + u[1] - 2.0;
            r[1] = u[0] + u[1] - 2.0 + 1e-13 * (u[1] - 1.0);
        }
        fn jacobian(&self, _u: &[f64], _lambda: &[f64], j: &mut bifurcata::matrix::Matrix) {
            *j = bifurcata::matrix::Matrix::from_row_major(2, 2, &[1.0, 1.0, 1.0, 1.0 + 1e-13]);
        }
    }
    let sys = EquilibriumSystem::new(&IllConditioned, &[0.0]);
    let mut x = vec![5.0, -1.0];
    let report = NewtonSolver::new(NewtonOptions { max_iterations: 100, ..Default::default() }).solve(&sys, &mut x).expect("converges");
    assert!(report.used_levenberg, "the ill-conditioned Jacobian sends it to Levenberg–Marquardt");
    close(x[0] + x[1], 2.0, 1e-9, "on the solution line u + v = 2");
}

#[test]
fn input_validation() {
    let sys = EquilibriumSystem::new(&Brusselator, &[1.0, 2.0]);
    let mut x = vec![1.0];
    let failure = NewtonSolver::default().solve(&sys, &mut x).expect_err("a mis-sized initial guess is rejected");
    assert_eq!(failure.code, SolveCode::InvalidInput, "and is classified as invalid input");

    let mut newton = NewtonSolver::default();
    newton.set_typical_f(vec![1.0]);
    let mut x = vec![1.0, 2.0];
    let failure = newton.solve(&sys, &mut x).expect_err("a mis-sized typical_f is rejected");
    assert_eq!(failure.code, SolveCode::InvalidInput);
}
