//! The equilibrium defining system and its test functions (specification
//! §6.1–6.3).
//!
//! The unknown vector packs the reduced state and the active parameter,
//! `x = (u₀ … u_{n−1}, λ_active)`, N = n + 1, and `G(x) = F(u, λ)` maps ℝᴺ to
//! ℝⁿ, so the solution set is a curve. Every other parameter is held at the
//! value supplied at construction.
//!
//! Test functions (§6.3):
//!
//! ```text
//! ψ_LP = τ_λ               the tangent's parameter component
//! ψ_BP = det[DG; τᵀ]       normalised; changes sign at a BP, not at an LP
//! ψ_H  = det(2F_u ⊙ I)     normalised bialternate determinant
//! ```
//!
//! ψ_H vanishes for a Hopf pair **and** for a neutral saddle (two real
//! eigenvalues summing to zero). [`process_singularity`] tells them apart from
//! the spectrum and records a neutral saddle as such rather than discarding
//! the zero, so a test-function crossing is never silently lost.
//!
//! Normal-form coefficients are milestone M3, so every record here has
//! confidence `spectrum`.
//!
//! [`process_singularity`]: DefiningSystem::process_singularity

use crate::bialternate::{hopf_test_function, is_hopf_spectrum, is_neutral_saddle_spectrum};
use crate::continuation::{ContinuationOptions, DefiningSystem};
use crate::linalg::{eigenvalues, normalised_determinant, unstable_dimension};
use crate::matrix::Matrix;
use crate::problem::BifurcationProblem;
use crate::types::*;

/// Test-function indices.
pub const TF_FOLD: usize = 0;
pub const TF_BRANCH_POINT: usize = 1;
pub const TF_HOPF: usize = 2;

pub struct EquilibriumCurve<P: BifurcationProblem> {
    problem: P,
    /// The full parameter vector; the active slot is overwritten from x.
    lambda: Vec<f64>,
    active: usize,
    n: usize,
    eigenvalue_zero_tol: f64,
}

impl<P: BifurcationProblem> EquilibriumCurve<P> {
    /// Continue `problem` in parameter `active`, the others held at `lambda0`.
    pub fn new(problem: P, lambda0: &[f64], active: usize, options: &ContinuationOptions) -> Result<Self, SolveStatus> {
        if active >= problem.parameter_count() {
            return Err(SolveStatus::fail(
                SolveCode::InvalidInput,
                format!("equilibrium curve: active parameter {active} is out of range (the problem has {})", problem.parameter_count()),
            ));
        }
        if lambda0.len() != problem.parameter_count() {
            return Err(SolveStatus::fail(
                SolveCode::InvalidInput,
                format!("equilibrium curve: lambda has {} entries, the problem has {} parameters", lambda0.len(), problem.parameter_count()),
            ));
        }
        let n = problem.state_dim();
        Ok(Self { problem, lambda: lambda0.to_vec(), active, n, eigenvalue_zero_tol: options.eigenvalue_zero_tol })
    }

    pub fn problem(&self) -> &P {
        &self.problem
    }

    pub fn state_dim(&self) -> usize {
        self.n
    }

    /// The index of the active parameter in the problem's parameter list.
    pub fn active_parameter(&self) -> usize {
        self.active
    }

    /// x from a state and a value of the active parameter.
    pub fn pack(&self, u: &[f64], parameter: f64) -> Vec<f64> {
        assert_eq!(u.len(), self.n, "pack: the state has {} entries, expected {}", u.len(), self.n);
        let mut x = u.to_vec();
        x.push(parameter);
        x
    }

    /// The full parameter vector at x.
    pub fn lambda_at(&self, x: &[f64]) -> Vec<f64> {
        let mut lambda = self.lambda.clone();
        lambda[self.active] = x[self.n];
        lambda
    }

    /// `F_u` at x.
    pub fn state_jacobian(&self, x: &[f64]) -> Matrix {
        let mut fu = Matrix::zeros(self.n, self.n);
        self.problem.jacobian(&x[..self.n], &self.lambda_at(x), &mut fu);
        fu
    }
}

impl<P: BifurcationProblem> DefiningSystem for EquilibriumCurve<P> {
    fn system_dim(&self) -> usize {
        self.n + 1
    }

    fn active_parameter_index(&self) -> usize {
        // Within x, the active parameter is the last component.
        self.n
    }

    fn residual(&self, x: &[f64], g: &mut [f64]) {
        self.problem.residual(&x[..self.n], &self.lambda_at(x), g);
    }

    fn jacobian(&self, x: &[f64], dg: &mut Matrix) {
        let n = self.n;
        let lambda = self.lambda_at(x);
        let mut fu = Matrix::zeros(n, n);
        self.problem.jacobian(&x[..n], &lambda, &mut fu);
        let mut fp = vec![0.0; n];
        self.problem.parameter_derivative(&x[..n], &lambda, self.active, &mut fp);
        *dg = Matrix::from_fn(n, n + 1, |i, j| if j < n { fu[(i, j)] } else { fp[i] });
    }

    fn test_function_count(&self) -> usize {
        3
    }

    fn test_function_kind(&self, i: usize) -> BifurcationKind {
        match i {
            TF_FOLD => BifurcationKind::Fold,
            TF_BRANCH_POINT => BifurcationKind::BranchPoint,
            TF_HOPF => BifurcationKind::Hopf,
            _ => BifurcationKind::None,
        }
    }

    fn test_functions(&self, x: &[f64], tau: &[f64], psi: &mut [f64]) {
        let n = self.n;
        // LP: zero at a fold, with a clean sign change because the tangent is
        // continuous along the branch.
        psi[TF_FOLD] = tau[n];

        // BP: nonzero at a fold — F_u is singular there but [F_u F_λ] still has
        // full rank — and zero at a branch point, where it does not.
        let mut dg = Matrix::zeros(n, n + 1);
        self.jacobian(x, &mut dg);
        let bordered = Matrix::from_fn(n + 1, n + 1, |i, j| if i < n { dg[(i, j)] } else { tau[j] });
        psi[TF_BRANCH_POINT] = normalised_determinant(&bordered);

        let fu = Matrix::from_fn(n, n, |i, j| dg[(i, j)]);
        psi[TF_HOPF] = hopf_test_function(&fu);
    }

    fn process_singularity(&self, i: usize, x: &[f64], _tau: &[f64]) -> Result<BifurcationInfo, String> {
        let values = eigenvalues(&self.state_jacobian(x)).map_err(|e| format!("the eigenvalue computation failed at the located point: {}", e.detail))?;
        let tol = self.eigenvalue_zero_tol.sqrt();
        let mut info = BifurcationInfo { point: self.decode(x), confidence: Confidence::Spectrum, ..Default::default() };

        match i {
            TF_FOLD => {
                // Confirm a real eigenvalue really is at zero. If none is, the
                // tangent's parameter component passed through zero without F_u
                // becoming singular — report that rather than a fold that is
                // not there.
                let smallest = values.iter().map(|z| z.modulus()).fold(f64::INFINITY, f64::min);
                if smallest > tol {
                    return Err(format!(
                        "the tangent's parameter component vanished but the smallest |eigenvalue| is {smallest:.3e}, so F_u is not singular here"
                    ));
                }
                info.kind = BifurcationKind::Fold;
                info.detail = format!("fold: smallest |eigenvalue| = {smallest:.3e}");
            }
            TF_BRANCH_POINT => {
                info.kind = BifurcationKind::BranchPoint;
                info.detail = "branch point: the tangent-bordered Jacobian is singular".to_owned();
            }
            TF_HOPF => {
                if let Some(omega) = is_hopf_spectrum(&values, tol) {
                    info.kind = BifurcationKind::Hopf;
                    info.normal_form.set("omega", omega);
                    info.detail = format!("Hopf: omega = {omega:.6}");
                } else if is_neutral_saddle_spectrum(&values, tol) {
                    // §10.3: a record of its own kind, so that a test-function
                    // zero is never silently discarded.
                    info.kind = BifurcationKind::NeutralSaddle;
                    info.detail = "neutral saddle: a real eigenvalue pair sums to zero, which is not a bifurcation".to_owned();
                } else {
                    info.kind = BifurcationKind::NeutralSaddle;
                    info.detail = "the bialternate determinant crossed zero but the spectrum shows neither a Hopf pair nor a real pair \
                                   summing to zero; recorded unclassified"
                        .to_owned();
                    info.confidence = Confidence::Degenerate;
                }
            }
            _ => return Err(format!("no test function {i}")),
        }
        Ok(info)
    }

    /// The point, with the spectrum of F_u (§6.2): verbose but cheap at these
    /// dimensions, and what lets the comparator say why a run diverged from a
    /// baseline rather than only that it did.
    fn decode(&self, x: &[f64]) -> CurvePoint {
        let mut p = CurvePoint { u: x[..self.n].to_vec(), lambda: self.lambda_at(x), ..Default::default() };
        if let Ok(values) = eigenvalues(&self.state_jacobian(x)) {
            p.unstable_dim = unstable_dimension(&values, 0.0);
            p.eigenvalues = values;
        }
        p
    }
}
