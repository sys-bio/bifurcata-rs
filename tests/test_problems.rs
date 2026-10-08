//! The hand-written test problems underpin most later suites, so their
//! analytic Jacobians, parameter derivatives and closed-form facts are checked
//! here first — against finite differences and against our own eigensolver.

use bifurcata::linalg::eigenvalues;
use bifurcata::matrix::{Matrix, norm_inf};
use bifurcata::problem::{BifurcationProblem, fd_parameter_derivative, numerical_jacobian};
use bifurcata::test_problems::*;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!((actual - expected).abs() <= tol, "{what}: got {actual}, expected {expected} (tolerance {tol:e})");
}

/// The analytic Jacobian agrees with central differences at `u`.
fn check_jacobian(problem: &dyn BifurcationProblem, u: &[f64], lambda: &[f64], name: &str) {
    let mut j = Matrix::zeros(0, 0);
    problem.jacobian(u, lambda, &mut j);
    let numerical = numerical_jacobian(problem, u, lambda);
    let n = problem.state_dim();
    assert_eq!((j.rows(), j.cols()), (n, n), "{name}: Jacobian is n×n");
    for r in 0..n {
        for c in 0..n {
            let scale = j[(r, c)].abs().max(1.0);
            close(j[(r, c)], numerical[(r, c)], 1e-6 * scale, &format!("{name}: J[{r}][{c}]"));
        }
    }
}

/// An analytic parameter derivative agrees with the finite-difference one.
fn check_parameter_derivative(problem: &dyn BifurcationProblem, u: &[f64], lambda: &[f64], name: &str) {
    let n = problem.state_dim();
    for k in 0..problem.parameter_count() {
        let (mut analytic, mut fd) = (vec![0.0; n], vec![0.0; n]);
        problem.parameter_derivative(u, lambda, k, &mut analytic);
        fd_parameter_derivative(problem, u, lambda, k, &mut fd);
        for i in 0..n {
            close(analytic[i], fd[i], 1e-6 * analytic[i].abs().max(1.0), &format!("{name}: dF{i}/dλ{k}"));
        }
    }
}

#[test]
fn analytic_jacobians_match_finite_differences() {
    // Points away from anything special, so no entry is accidentally zero.
    check_jacobian(&Brusselator, &[1.3, 2.4], &[1.0, 2.7], "Brusselator");
    check_jacobian(&FoldProblem, &[0.7], &[0.3], "fold");
    check_jacobian(&BadlyScaledProblem { scale_u: 1e3, scale_v: 1e-3, target_u: 1.0, target_v: 2.0 }, &[0.5, 0.5], &[0.0], "badly scaled");
    check_jacobian(&GoodwinProblem::new(5), &[0.4, 0.5, 0.6, 0.7, 0.8], &[9.0], "Goodwin n = 5");
    check_jacobian(&TranscriticalProblem, &[0.4], &[1.1], "transcritical");
    check_jacobian(&ArctanProblem, &[2.0], &[0.5], "arctan");
    check_jacobian(&SingularConsistentProblem, &[0.2, 0.3], &[0.0], "singular consistent");
    check_jacobian(&CubicProblem, &[0.6], &[0.2], "cubic");
    check_jacobian(&NeutralSaddleProblem, &[0.4, 0.3], &[0.7], "neutral saddle");
    check_jacobian(&Selkov, &[1.3, 0.4], &[0.5, 0.6], "Selkov");
}

#[test]
fn analytic_parameter_derivatives_match_finite_differences() {
    check_parameter_derivative(&Brusselator, &[1.3, 2.4], &[1.0, 2.7], "Brusselator");
    check_parameter_derivative(&TranscriticalProblem, &[0.4], &[1.1], "transcritical");
}

#[test]
fn brusselator_closed_forms() {
    let (a, b) = (1.0, 2.0);
    let u = Brusselator::equilibrium(a, b);
    let mut r = [0.0; 2];
    Brusselator.residual(&u, &[a, b], &mut r);
    close(norm_inf(&r), 0.0, 1e-14, "the residual vanishes at the analytic equilibrium");

    // Jacobian there [[B−1, A²], [−B, −A²]].
    let mut j = Matrix::zeros(0, 0);
    Brusselator.jacobian(&u, &[a, b], &mut j);
    let exact = [[b - 1.0, a * a], [-b, -a * a]];
    for (row, values) in exact.iter().enumerate() {
        for (col, value) in values.iter().enumerate() {
            close(j[(row, col)], *value, 1e-14, &format!("J[{row}][{col}] at the equilibrium"));
        }
    }

    // At the Hopf, B = 1 + A², the eigenvalues are ±iA.
    let b = Brusselator::hopf_b(a);
    let mut j = Matrix::zeros(0, 0);
    Brusselator.jacobian(&Brusselator::equilibrium(a, b), &[a, b], &mut j);
    let ev = eigenvalues(&j).unwrap();
    close(ev[0].re, 0.0, 1e-14, "Hopf: zero real part");
    close(ev[0].im.abs(), Brusselator::hopf_omega(a), 1e-14, "Hopf: ω = A");
}

/// The Goodwin Hopf, through our own eigensolver: at the critical Hill
/// coefficient the uniform steady state's Jacobian has eigenvalues ±i tan(π/n).
#[test]
fn goodwin_hopf_is_where_the_closed_form_says() {
    close(GoodwinProblem::critical_g(3), 8.0, 1e-12, "n = 3 needs g = sec(60°)³ = 8");
    close(GoodwinProblem::hopf_omega(3), 3f64.sqrt(), 1e-14, "n = 3: ω = tan(60°) = √3");

    for n in [3, 5, 10] {
        let p = GoodwinProblem::critical_p(n).unwrap();
        let x = GoodwinProblem::steady_state_value(p);
        let problem = GoodwinProblem::new(n);
        let u = vec![x; n];
        let mut r = vec![0.0; n];
        problem.residual(&u, &[p], &mut r);
        close(norm_inf(&r), 0.0, 1e-12, &format!("n = {n}: the uniform state is a steady state"));

        let mut j = Matrix::zeros(0, 0);
        problem.jacobian(&u, &[p], &mut j);
        let ev = eigenvalues(&j).unwrap();
        // The pair with the largest real part is the one that crosses. (The
        // eigenvalues come in the solver's order, conjugate pairs adjacent, as
        // LAPACK returns them — not sorted.)
        let critical = ev.iter().copied().max_by(|a, b| a.re.total_cmp(&b.re)).unwrap();
        close(critical.re, 0.0, 1e-9, &format!("n = {n}: the critical pair is on the imaginary axis"));
        // (Only this pair is checked for ω: the roots lie on a circle around
        // −1, so another pair can share its imaginary part — at n = 10,
        // −2 ± 0.3249i does.)
        close(critical.im.abs(), GoodwinProblem::hopf_omega(n), 1e-9, &format!("n = {n}: ω = tan(π/n)"));
    }
}
