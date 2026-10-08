//! Damped Newton with an Armijo line search and a Levenberg–Marquardt
//! fallback (specification §5.3, §6.1).
//!
//! There is exactly **one** Newton in the library (§2.1): the continuation
//! corrector is this solver handed a different [`NonlinearSystem`] (the
//! pseudo-arclength system), and the initial-point solve is the same code on
//! the model at a fixed parameter ([`EquilibriumSystem`]).
//!
//! **Scaling** (Dennis & Schnabel ch. 7): metabolic models span µM to mM, so an
//! unscaled convergence test means nothing across that range. Two vectors of
//! typical values do the scaling:
//!
//! * `typical_x[i]` — the magnitude below which a change in `x[i]` doesn't matter;
//! * `typical_f[i]` — the magnitude below which a residual in equation `i` is negligible.
//!
//! Convergence is `max |F_i| / typical_f_i <= tol_residual`, and the line
//! search uses the same scaled quantity, `½ Σ (F_i / typical_f_i)²`, so the two
//! cannot disagree about what progress means. `typical_f` defaults to 1 — an
//! absolute test, which calls a model whose rates are all ~1e-9 converged
//! before it starts — so a real model should set it:
//! [`NewtonSolver::scale_from_jacobian`] is the recommended route,
//! [`NewtonSolver::typical_f_from_residual`] the Delphi version's.
//!
//! Ported from websim's port of the Delphi `Bifurcata.Newton`, which carries
//! the M0 fixes: Levenberg–Marquardt damping relative to the scale of JᵀJ, a
//! direct Levenberg–Marquardt step when the line search stalls, and
//! [`NonlinearSystem::admissible`].

use crate::linalg::Lu;
use crate::matrix::Matrix;
use crate::problem::BifurcationProblem;
use crate::types::{SolveCode, SolveStatus};

/// A square nonlinear system `F(x) = 0`.
pub trait NonlinearSystem {
    fn dim(&self) -> usize;

    fn residual(&self, x: &[f64], r: &mut [f64]);

    /// `∂F/∂x`, resized to dim × dim.
    fn jacobian(&self, x: &[f64], j: &mut Matrix);

    /// Whether `x` is allowed at all. The line search backs away from any
    /// trial point that is not, as if its merit were infinite — which is how a
    /// solve is kept inside the physical region of a reaction network, where
    /// the poles of the rate laws lie on the far side of zero.
    fn admissible(&self, _x: &[f64]) -> bool {
        true
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NewtonOptions {
    pub max_iterations: usize,
    pub tol_residual: f64,
    pub tol_step: f64,
    pub use_line_search: bool,
    pub max_line_search_steps: usize,
    /// The sufficient-decrease constant of the Armijo condition.
    pub armijo_alpha: f64,
    /// Clamps on the backtracking factor from the quadratic model.
    pub min_backtrack: f64,
    pub max_backtrack: f64,
    pub use_levenberg_fallback: bool,
    /// Below this reciprocal condition number, take a Levenberg–Marquardt step.
    pub rcond_threshold: f64,
    pub lm_initial_mu: f64,
    pub lm_increase: f64,
    pub lm_decrease: f64,
    pub lm_max_mu: f64,
    /// Give up if the merit function grows by this factor.
    pub divergence_factor: f64,
}

impl Default for NewtonOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tol_residual: 1e-10,
            tol_step: 1e-10,
            use_line_search: true,
            max_line_search_steps: 25,
            armijo_alpha: 1e-4,
            min_backtrack: 0.1,
            max_backtrack: 0.5,
            use_levenberg_fallback: true,
            rcond_threshold: 1e-12,
            lm_initial_mu: 1e-3,
            lm_increase: 10.0,
            lm_decrease: 0.1,
            lm_max_mu: 1e12,
            divergence_factor: 1e8,
        }
    }
}

/// A successful solve. A failure is a [`SolveStatus`] whose `iterations` and
/// `residual_norm` say how far it got.
#[derive(Clone, Debug, PartialEq)]
pub struct NewtonReport {
    pub iterations: usize,
    /// The scaled residual (max norm) at the solution.
    pub residual_norm: f64,
    /// Whether the accepted steps included a Levenberg–Marquardt or
    /// steepest-descent step.
    pub used_levenberg: bool,
}

#[derive(Clone, Debug, Default)]
pub struct NewtonSolver {
    pub options: NewtonOptions,
    typical_x: Option<Vec<f64>>,
    typical_f: Option<Vec<f64>>,
}

impl NewtonSolver {
    pub fn new(options: NewtonOptions) -> Self {
        Self { options, typical_x: None, typical_f: None }
    }

    pub fn set_typical_x(&mut self, v: Vec<f64>) {
        self.typical_x = Some(v);
    }

    pub fn set_typical_f(&mut self, v: Vec<f64>) {
        self.typical_f = Some(v);
    }

    pub fn clear_typicals(&mut self) {
        self.typical_x = None;
        self.typical_f = None;
    }

    /// Set `typical_f` from `|F(x0)|`, each entry floored at `floor` so an
    /// equation already at zero does not make itself untestable — the Delphi
    /// version's scaling, and right for a problem whose equations differ in
    /// scale by many orders but none of which starts at zero.
    ///
    /// For a reaction network prefer [`scale_from_jacobian`]: a rate that
    /// happens to be exactly zero at the start gets a typical size of `floor`,
    /// its row is scaled up by 1/`floor`, and it swamps every other equation.
    ///
    /// [`scale_from_jacobian`]: NewtonSolver::scale_from_jacobian
    pub fn typical_f_from_residual(&mut self, system: &dyn NonlinearSystem, x0: &[f64], floor: f64) {
        let mut r = vec![0.0; system.dim()];
        system.residual(x0, &mut r);
        self.typical_f = Some(r.iter().map(|v| v.abs().max(floor)).collect());
    }

    /// Set `typical_x` from the magnitudes in `x0` and `typical_f` from the
    /// Jacobian there: `typical_f[i] = Σ_j |J_ij| typical_x[j]`, how much
    /// equation i's residual changes for a typical change in the state.
    ///
    /// It makes the convergence test mean what we want — a scaled residual of
    /// 1e-10 is a state error of about 1e-10 of its typical size — and, unlike
    /// scaling by `|F(x0)|`, it doesn't break when a rate happens to be zero at
    /// the starting point (common when a model starts from zero).
    ///
    /// `typical_x[i]` is `|x0_i|`, floored at a thousandth of the largest
    /// `|x0_j|`, or 1 if the state is all zero.
    pub fn scale_from_jacobian(&mut self, system: &dyn NonlinearSystem, x0: &[f64]) {
        let largest = x0.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let typical_x: Vec<f64> = x0
            .iter()
            .map(|v| {
                let t = v.abs().max(1e-3 * largest);
                if t > 0.0 { t } else { 1.0 }
            })
            .collect();
        let mut j = Matrix::zeros(0, 0);
        system.jacobian(x0, &mut j);
        let mut typical_f: Vec<f64> = (0..j.rows())
            .map(|i| (0..j.cols()).map(|c| j[(i, c)].abs() * typical_x[c]).sum::<f64>())
            .map(|v: f64| if v.is_finite() { v } else { 0.0 })
            .collect();
        // An equation with no dependence on the state at x0 still needs a scale.
        let biggest = typical_f.iter().fold(0.0f64, |m, v| m.max(*v));
        let floor = (1e-6 * biggest).max(1e-300);
        for v in &mut typical_f {
            *v = v.max(floor);
        }
        self.typical_x = Some(typical_x);
        self.typical_f = Some(typical_f);
    }

    /// Solve `F(x) = 0`, starting from and updating `x`. On failure `x` holds
    /// the last accepted iterate.
    pub fn solve(&self, system: &dyn NonlinearSystem, x: &mut [f64]) -> Result<NewtonReport, SolveStatus> {
        let n = system.dim();
        if n == 0 {
            return Err(SolveStatus::fail(SolveCode::InvalidInput, "Newton: the system dimension is 0"));
        }
        if x.len() != n {
            return Err(SolveStatus::fail(
                SolveCode::InvalidInput,
                format!("Newton: the initial guess has length {}, the system dimension is {n}", x.len()),
            ));
        }
        let o = &self.options;
        let typical_f = match &self.typical_f {
            Some(v) if v.len() == n => v.clone(),
            Some(v) => {
                return Err(SolveStatus::fail(SolveCode::InvalidInput, format!("Newton: typical_f has length {}, the system dimension is {n}", v.len())));
            }
            None => vec![1.0; n],
        };
        let typical_x = match &self.typical_x {
            Some(v) if v.len() == n => v.clone(),
            Some(v) => {
                return Err(SolveStatus::fail(SolveCode::InvalidInput, format!("Newton: typical_x has length {}, the system dimension is {n}", v.len())));
            }
            None => x.iter().map(|v| v.abs().max(1.0)).collect(),
        };

        let scale = |r: &[f64]| -> Vec<f64> { r.iter().zip(&typical_f).map(|(a, t)| a / t).collect() };
        let merit = |rs: &[f64]| 0.5 * rs.iter().map(|v| v * v).sum::<f64>();
        let max_abs = |v: &[f64]| v.iter().fold(0.0f64, |m, a| m.max(a.abs()));
        let fail = |code: SolveCode, detail: String, iterations: usize, residual: f64| {
            Err(SolveStatus { code, detail, iterations, residual_norm: residual })
        };

        let mut r = vec![0.0; n];
        system.residual(x, &mut r);
        let mut rs = scale(&r);
        let mut f = merit(&rs);
        let initial_merit = f;
        // Already there. Checked before the first factorisation, so a converged
        // start costs one residual — and a zero residual at a singular point is
        // converged, not a factorisation failure.
        if max_abs(&rs) <= o.tol_residual {
            return Ok(NewtonReport { iterations: 0, residual_norm: max_abs(&rs), used_levenberg: false });
        }

        let mut mu = o.lm_initial_mu;
        let mut used_levenberg = false;
        let mut trial = vec![0.0; n];
        let mut trial_r = vec![0.0; n];
        let mut j = Matrix::zeros(n, n);

        for iteration in 1..=o.max_iterations {
            // Row i of J is divided by typical_f[i], matching the residual
            // scaling, so the Newton step is unchanged while the merit
            // function and its gradient stay consistent.
            system.jacobian(x, &mut j);
            for col in 0..n {
                for row in 0..n {
                    j[(row, col)] /= typical_f[row];
                }
            }

            // The Newton step, unless J is singular or too ill-conditioned to trust.
            let lu = Lu::factor(&j);
            let well_conditioned = !lu.singular() && (!o.use_levenberg_fallback || lu.reciprocal_condition() >= o.rcond_threshold);
            let mut lm_step_taken = false;
            let mut step = if well_conditioned {
                lu.solve(&rs.iter().map(|v| -v).collect::<Vec<_>>())?
            } else if o.use_levenberg_fallback {
                lm_step_taken = true;
                levenberg_marquardt_step(&j, &rs, mu).map_err(|e| SolveStatus {
                    code: SolveCode::SingularJacobian,
                    detail: format!("Newton: the Levenberg–Marquardt system is singular at iteration {iteration} with mu = {mu:.3e} ({})", e.detail),
                    iterations: iteration,
                    residual_norm: max_abs(&rs),
                })?
            } else {
                return fail(
                    SolveCode::SingularJacobian,
                    format!("Newton: the Jacobian is singular at iteration {iteration} and the Levenberg–Marquardt fallback is disabled"),
                    iteration,
                    max_abs(&rs),
                );
            };

            // The merit's slope along the step, from the gradient Jᵀ rs: the
            // identity slope = −2f holds only for an exact Newton step.
            let gradient = j.mul_transpose_vec(&rs);
            let mut slope: f64 = gradient.iter().zip(&step).map(|(g, s)| g * s).sum();
            if slope >= 0.0 {
                // Not a descent direction (a badly polluted solve): steepest descent.
                step = gradient.iter().map(|g| -g).collect();
                slope = gradient.iter().zip(&step).map(|(g, s)| g * s).sum();
                lm_step_taken = true;
                if slope >= 0.0 {
                    return fail(SolveCode::SingularJacobian, format!("Newton: no descent direction at iteration {iteration}"), iteration, max_abs(&rs));
                }
            }

            // Armijo backtracking with a quadratic model of the merit (§5.3).
            let mut factor = 1.0;
            let mut accepted = false;
            let mut trial_merit = f;
            let steps = if o.use_line_search { o.max_line_search_steps } else { 1 };
            for _ in 0..steps {
                for k in 0..n {
                    trial[k] = x[k] + factor * step[k];
                }
                if !system.admissible(&trial) {
                    factor *= o.max_backtrack;
                    continue;
                }
                system.residual(&trial, &mut trial_r);
                let trs = scale(&trial_r);
                trial_merit = merit(&trs);
                if trial_merit.is_finite() && (!o.use_line_search || trial_merit <= f + o.armijo_alpha * factor * slope) {
                    accepted = true;
                    rs = trs;
                    r.copy_from_slice(&trial_r);
                    break;
                }
                let denominator = 2.0 * (trial_merit - f - factor * slope);
                let new_factor = if trial_merit.is_finite() && denominator > 0.0 { -slope * factor * factor / denominator } else { factor * o.max_backtrack };
                factor = new_factor.clamp(o.min_backtrack * factor, o.max_backtrack * factor);
            }

            if !accepted {
                if o.use_levenberg_fallback && mu < o.lm_max_mu {
                    // The line search stalled: lean harder on Levenberg–Marquardt
                    // and take its step directly (§6.1 prescribes raising mu
                    // rather than giving up).
                    mu = (mu * o.lm_increase).min(o.lm_max_mu);
                    if let Ok(lm) = levenberg_marquardt_step(&j, &rs, mu) {
                        for k in 0..n {
                            trial[k] = x[k] + lm[k];
                        }
                        if system.admissible(&trial) {
                            system.residual(&trial, &mut trial_r);
                            let trs = scale(&trial_r);
                            let m = merit(&trs);
                            if m.is_finite() && m < f {
                                used_levenberg = true;
                                let scaled_step = scaled_step_norm(&lm, x, &typical_x);
                                x.copy_from_slice(&trial);
                                r.copy_from_slice(&trial_r);
                                rs = trs;
                                f = m;
                                if max_abs(&rs) <= o.tol_residual || (scaled_step <= o.tol_step && max_abs(&rs) <= o.tol_residual.sqrt()) {
                                    return Ok(NewtonReport { iterations: iteration, residual_norm: max_abs(&rs), used_levenberg });
                                }
                                continue;
                            }
                        }
                    }
                }
                return fail(
                    SolveCode::LineSearchFailed,
                    format!("Newton: the line search failed to find a decrease at iteration {iteration} (merit {f:.6e}, slope {slope:.6e})"),
                    iteration,
                    max_abs(&rs),
                );
            }

            let applied: Vec<f64> = step.iter().map(|s| s * factor).collect();
            let scaled_step = scaled_step_norm(&applied, x, &typical_x);
            x.copy_from_slice(&trial);
            f = trial_merit;
            if lm_step_taken {
                used_levenberg = true;
                mu = (mu * o.lm_decrease).max(o.lm_initial_mu * 1e-6);
            }

            let residual = max_abs(&rs);
            if residual <= o.tol_residual {
                return Ok(NewtonReport { iterations: iteration, residual_norm: residual, used_levenberg });
            }
            // The step test fires when the iterate has stopped moving. That is
            // convergence only if the residual is also small; otherwise the
            // solver is stuck, which is a different outcome and is reported
            // as one.
            if scaled_step <= o.tol_step {
                if residual <= o.tol_residual.sqrt() {
                    return Ok(NewtonReport { iterations: iteration, residual_norm: residual, used_levenberg });
                }
                return fail(
                    SolveCode::NotConverged,
                    format!(
                        "Newton: the step collapsed to {scaled_step:.3e} at iteration {iteration} but the scaled residual is still {residual:.3e}; \
                         the iterate is stuck, not converged"
                    ),
                    iteration,
                    residual,
                );
            }
            if f > o.divergence_factor * initial_merit {
                return fail(SolveCode::Diverged, format!("Newton: the merit grew from {initial_merit:.3e} to {f:.3e} by iteration {iteration}"), iteration, residual);
            }
        }
        fail(
            SolveCode::MaxIterations,
            format!("Newton: {} iterations exhausted, scaled residual {:.3e} (tolerance {:.3e})", o.max_iterations, max_abs(&rs), o.tol_residual),
            o.max_iterations,
            max_abs(&rs),
        )
    }
}

/// `max_i |dx_i| / max(|x_i|, typical_x_i)`
fn scaled_step_norm(dx: &[f64], x: &[f64], typical_x: &[f64]) -> f64 {
    dx.iter().zip(x).zip(typical_x).map(|((d, xi), t)| d.abs() / xi.abs().max(*t)).fold(0.0, f64::max)
}

/// Solve `(Jᵀ J + µ d I) δ = −Jᵀ F` (Dennis & Schnabel ch. 6; Kelley ch. 8),
/// where `d` is the largest diagonal entry of `Jᵀ J`, so that µ is a relative
/// damping that means the same whatever the scale of J. Forming the normal
/// matrix squares the condition number, which is acceptable because this path
/// is taken only when J is already too ill-conditioned to use directly — the
/// damping is what restores solvability.
fn levenberg_marquardt_step(j: &Matrix, rs: &[f64], mu: f64) -> Result<Vec<f64>, SolveStatus> {
    let n = j.cols();
    let mut normal = j.transpose().mul(j);
    let largest_diagonal = (0..n).map(|i| normal[(i, i)]).fold(0.0f64, f64::max);
    let damping = mu * if largest_diagonal > 0.0 && largest_diagonal.is_finite() { largest_diagonal } else { 1.0 };
    normal.add_scaled_identity(damping);
    let rhs: Vec<f64> = j.mul_transpose_vec(rs).iter().map(|v| -v).collect();
    Lu::new(&normal)?.solve(&rhs)
}

/// A model at fixed parameters, `F(u) = F(u, λ)`, as a [`NonlinearSystem`]:
/// the initial-point solve of §6.1.
pub struct EquilibriumSystem<'a, P: BifurcationProblem + ?Sized> {
    pub problem: &'a P,
    pub lambda: Vec<f64>,
}

impl<'a, P: BifurcationProblem + ?Sized> EquilibriumSystem<'a, P> {
    pub fn new(problem: &'a P, lambda: &[f64]) -> Self {
        Self { problem, lambda: lambda.to_vec() }
    }
}

impl<P: BifurcationProblem + ?Sized> NonlinearSystem for EquilibriumSystem<'_, P> {
    fn dim(&self) -> usize {
        self.problem.state_dim()
    }
    fn residual(&self, x: &[f64], r: &mut [f64]) {
        self.problem.residual(x, &self.lambda, r);
    }
    fn jacobian(&self, x: &[f64], j: &mut Matrix) {
        self.problem.jacobian(x, &self.lambda, j);
    }
}
