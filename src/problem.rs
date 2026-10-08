//! The model contract (specification §3.1).
//!
//! Everything the library needs from an ODE system `du/dt = F(u, λ)`,
//! `u ∈ ℝⁿ`, `λ ∈ ℝᵖ`, is expressed by [`BifurcationProblem`]. The Antimony
//! models of websim implement it (behind the `antimony` feature, milestone M2);
//! tests supply hand-written ones ([`crate::test_problems`]).
//!
//! The state `u` is the **reduced** state — independent species only (§3.2).
//! For a network with conserved moieties the full Jacobian is structurally
//! singular, so this is a hard requirement, not an optimisation.

use crate::matrix::Matrix;

pub trait BifurcationProblem {
    /// n: the number of (independent) states.
    fn state_dim(&self) -> usize;

    /// p: the number of parameters.
    fn parameter_count(&self) -> usize;

    /// `r = F(u, λ)`; `r` has length n.
    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]);

    /// `J = ∂F/∂u`, n×n. Required, not optional (§3.1).
    fn jacobian(&self, u: &[f64], lambda: &[f64], j: &mut Matrix);

    /// `∂F/∂λ_k` as a column of length n. Finite differences by default: it
    /// appears once per continuation step, in the tangent, where a relative
    /// error of 1e-8 is immaterial (§3.1). Override when an analytic form is
    /// available and cheap.
    fn parameter_derivative(&self, u: &[f64], lambda: &[f64], k: usize, out: &mut [f64]) {
        fd_parameter_derivative(self, u, lambda, k, out);
    }

    fn state_name(&self, i: usize) -> String {
        format!("u{i}")
    }

    fn parameter_name(&self, k: usize) -> String {
        format!("p{k}")
    }
}

/// A reference to a problem is a problem, so a defining system can borrow one.
/// Every method forwards, overrides included.
impl<T: BifurcationProblem + ?Sized> BifurcationProblem for &T {
    fn state_dim(&self) -> usize {
        (**self).state_dim()
    }
    fn parameter_count(&self) -> usize {
        (**self).parameter_count()
    }
    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
        (**self).residual(u, lambda, r)
    }
    fn jacobian(&self, u: &[f64], lambda: &[f64], j: &mut Matrix) {
        (**self).jacobian(u, lambda, j)
    }
    fn parameter_derivative(&self, u: &[f64], lambda: &[f64], k: usize, out: &mut [f64]) {
        (**self).parameter_derivative(u, lambda, k, out)
    }
    fn state_name(&self, i: usize) -> String {
        (**self).state_name(i)
    }
    fn parameter_name(&self, k: usize) -> String {
        (**self).parameter_name(k)
    }
}

impl<T: BifurcationProblem + ?Sized> BifurcationProblem for Box<T> {
    fn state_dim(&self) -> usize {
        (**self).state_dim()
    }
    fn parameter_count(&self) -> usize {
        (**self).parameter_count()
    }
    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
        (**self).residual(u, lambda, r)
    }
    fn jacobian(&self, u: &[f64], lambda: &[f64], j: &mut Matrix) {
        (**self).jacobian(u, lambda, j)
    }
    fn parameter_derivative(&self, u: &[f64], lambda: &[f64], k: usize, out: &mut [f64]) {
        (**self).parameter_derivative(u, lambda, k, out)
    }
    fn state_name(&self, i: usize) -> String {
        (**self).state_name(i)
    }
    fn parameter_name(&self, k: usize) -> String {
        (**self).parameter_name(k)
    }
}

/// `∂F/∂λ_k` by central differences, with a step relative to `λ_k` (absolute
/// where it passes through zero, or the step would vanish with it). Usable
/// whether or not a problem overrides [`BifurcationProblem::parameter_derivative`],
/// so a test can hold an analytic derivative against it.
pub fn fd_parameter_derivative<P: BifurcationProblem + ?Sized>(problem: &P, u: &[f64], lambda: &[f64], k: usize, out: &mut [f64]) {
    let n = problem.state_dim();
    let h = 1e-7 * lambda[k].abs().max(1.0);
    let (mut plus, mut minus) = (lambda.to_vec(), lambda.to_vec());
    plus[k] += h;
    minus[k] -= h;
    // The step actually representable, so the divisor matches the perturbation.
    let h = (plus[k] - minus[k]) / 2.0;
    let (mut rp, mut rm) = (vec![0.0; n], vec![0.0; n]);
    problem.residual(u, &plus, &mut rp);
    problem.residual(u, &minus, &mut rm);
    for i in 0..n {
        out[i] = (rp[i] - rm[i]) / (2.0 * h);
    }
}

/// `∂F/∂u` by central differences, for validating an analytic Jacobian. Test
/// support, not for the solver path.
pub fn numerical_jacobian<P: BifurcationProblem + ?Sized>(problem: &P, u: &[f64], lambda: &[f64]) -> Matrix {
    let n = problem.state_dim();
    let mut j = Matrix::zeros(n, n);
    let (mut rp, mut rm) = (vec![0.0; n], vec![0.0; n]);
    for col in 0..n {
        let h = 1e-7 * u[col].abs().max(1.0);
        let (mut plus, mut minus) = (u.to_vec(), u.to_vec());
        plus[col] += h;
        minus[col] -= h;
        let h = (plus[col] - minus[col]) / 2.0;
        problem.residual(&plus, lambda, &mut rp);
        problem.residual(&minus, lambda, &mut rm);
        for row in 0..n {
            j[(row, col)] = (rp[row] - rm[row]) / (2.0 * h);
        }
    }
    j
}
