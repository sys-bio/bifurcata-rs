//! Dimension and scale (specification §1.3: 3–30 states is typical), ported
//! check by check from the Delphi `Bifurcata.Tests.Scale`.
//!
//! Not a performance test (§13.1 defers that). The question is narrower: does
//! the machinery still give the right answer as n grows? Three things scale
//! with n and could break — the bialternate product (1×1 at n = 2, 435×435 at
//! n = 30), eigenvector unpacking (more conjugate pairs to mis-pair), and the
//! bordered solves. The n-stage Goodwin oscillator is defined at any n and its
//! Hopf frequency is exactly tan(π/n), so correctness is asserted, not inspected.
//!
//! The Delphi suite's timing checks are not ported: wall-clock bounds make a
//! flaky test, and `std::time::Instant` is unavailable in the browser build.

use bifurcata::bialternate::{bialternate_dim, hopf_test_function};
use bifurcata::complex::{ZERO, cx};
use bifurcata::continuation::{ContinuationEngine, ContinuationOptions};
use bifurcata::equilibrium::EquilibriumCurve;
use bifurcata::linalg::eigen;
use bifurcata::matrix::Matrix;
use bifurcata::test_problems::GoodwinProblem;
use bifurcata::types::*;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!((actual - expected).abs() <= tol, "{what}: got {actual}, expected {expected} (tolerance {tol:e})");
}

#[test]
fn eigenvectors_at_higher_n() {
    // A circulant-like matrix, which reliably produces many complex pairs. The
    // residual is checked for EVERY eigenvector, so one mis-paired column shows.
    for n in [8, 16, 30] {
        let mut a = Matrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = -0.5;
            a[(i, (i + 1) % n)] = 1.0;
            a[(i, (i + n - 1) % n)] = -1.0;
        }
        let e = eigen(&a, true, true).unwrap_or_else(|s| panic!("n = {n}: eigen failed: {s}"));
        let pairs = e.values.iter().filter(|z| z.im > 1e-12).count();
        assert!(pairs >= 3, "n = {n}: the test matrix has {pairs} conjugate pairs to unpack");

        let mut worst = 0.0f64;
        for (j, mu) in e.values.iter().enumerate() {
            let (v, p) = (&e.right[j], &e.left[j]);
            for i in 0..n {
                // (A v)_i − μ v_i
                let av = (0..n).fold(ZERO, |s, k| s + cx(a[(i, k)], 0.0) * v[k]);
                worst = worst.max((av - *mu * v[i]).modulus());
                // (Aᵀ p)_i − conj(μ) p_i
                let atp = (0..n).fold(ZERO, |s, k| s + cx(a[(k, i)], 0.0) * p[k]);
                worst = worst.max((atp - mu.conjugate() * p[i]).modulus());
            }
        }
        close(worst, 0.0, 1e-11, &format!("n = {n}: every right AND left eigenvector satisfies its equation"));
    }
}

#[test]
fn bialternate_at_scale() {
    // §6.3: at n = 30 the determinant is a 435×435 LU at every point.
    for n in [10, 20, 30] {
        let mut a = Matrix::zeros(n, n);
        for k in 0..n {
            a[(k, k)] = -1.0 - k as f64 * 0.1;
            if k > 0 {
                a[(k, k - 1)] = 0.3;
            }
            if k + 1 < n {
                a[(k, k + 1)] = -0.2;
            }
        }
        let m = bialternate_dim(n);
        let psi = hopf_test_function(&a);
        assert!(psi.is_finite() && psi != 0.0, "n = {n}: the bialternate is {m}×{m} and the test function finite and nonzero ({psi})");
        // Every pairwise sum of a stable spectrum is negative, so the sign is (−1)^m.
        let expected_sign = if m.is_multiple_of(2) { 1.0 } else { -1.0 };
        assert_eq!(psi.signum(), expected_sign, "n = {n}: m = {m} negative pairwise sums give sign (−1)^m");
    }
}

/// Continue an n-stage Goodwin in the Hill coefficient across its Hopf and
/// check the located point against the analytic values.
fn goodwin_run(n: usize) {
    let p_crit = GoodwinProblem::critical_p(n).unwrap();
    let (p_lo, p_hi) = (0.6 * p_crit, 1.6 * p_crit);
    let options = ContinuationOptions { initial_step: 0.05, max_step: 0.2, max_points: 300, parameter_min: p_lo, parameter_max: p_hi, ..Default::default() };

    let curve = EquilibriumCurve::new(GoodwinProblem::new(n), &[p_lo], 0, &options).unwrap();
    // The steady state is uniform and known in closed form.
    let u0 = vec![GoodwinProblem::steady_state_value(p_lo); n];
    let x0 = curve.pack(&u0, p_lo);
    let mut engine = ContinuationEngine::new(curve, options);
    engine.initialise(&x0, 1).unwrap_or_else(|s| panic!("n = {n}: the branch failed to initialise: {s}"));
    for _ in 0..500 {
        if engine.step() != StepResult::Ok {
            break;
        }
    }
    let branch = engine.branch();
    let last = branch.last_point().unwrap();

    assert!(branch.points.len() > 10, "n = {n}: the branch produced {} points", branch.points.len());
    assert_eq!(last.u.len(), n, "n = {n}: points carry all n states");

    let hopf = branch.bifurcations.iter().find(|b| b.kind == BifurcationKind::Hopf).unwrap_or_else(|| panic!("n = {n}: the Hopf was detected"));
    // ω = tan(π/n) exactly, independent of p and of the steady state.
    close(hopf.normal_form.get_or("omega", 0.0), GoodwinProblem::hopf_omega(n), 1e-5, &format!("n = {n}: ω = tan(π/{n})"));
    close(hopf.point.lambda[0], p_crit, 1e-4, &format!("n = {n}: Hopf located at the analytic Hill coefficient {p_crit:.8}"));
    // A complex pair crossing: the unstable dimension changes by exactly 2.
    assert_eq!(branch.points[0].unstable_dim, 0, "n = {n}: stable below the Hopf");
    assert_eq!(last.unstable_dim, 2, "n = {n}: exactly two unstable directions above it");
    assert_eq!(branch.bifurcations.len(), 1, "n = {n}: nothing else reported on the branch");

    // The steady state stays uniform all the way along.
    let spread = last.u.iter().map(|v| (v - last.u[0]).abs()).fold(0.0, f64::max);
    close(spread, 0.0, 1e-7, &format!("n = {n}: the steady state stayed uniform"));
}

#[test]
fn goodwin_formulas() {
    // sec(60°)³ = 8: the textbook threshold for a 3-stage oscillator.
    close(GoodwinProblem::critical_g(3), 8.0, 1e-12, "n = 3 reproduces the classical Goodwin threshold g = 8");
    // n = 3 is not continued: g grows like ln p, so it needs p ≈ 24 000. The
    // sweep starts at 6, where the critical Hill coefficient is modest.
    let p6 = GoodwinProblem::critical_p(6).unwrap();
    assert!(p6 < 100.0, "n = 6 has a workable critical Hill coefficient ({p6:.3})");
}

#[test]
fn goodwin_n6() {
    goodwin_run(6);
}

#[test]
fn goodwin_n10() {
    goodwin_run(10);
}

#[test]
fn goodwin_n20() {
    goodwin_run(20);
}

