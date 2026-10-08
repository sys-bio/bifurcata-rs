//! Pseudo-arclength continuation (specification §4, §5).
//!
//! The engine is written once against [`DefiningSystem`] and knows nothing
//! about what it is continuing: an equilibrium branch, and later fold, Hopf
//! and periodic-orbit curves, all arrive as the same abstraction.
//!
//! **An iterator, not a loop** (§4.2): [`ContinuationEngine::step`] advances by
//! one point and returns, so a GUI can draw the branch as it grows and stop it
//! at any time — essential in the browser, which has no threads — and the
//! automatic diagram of §8 can suspend one branch while it explores another.
//!
//! Two deviations from the specification's interface, as in the Delphi
//! version: the test functions and `process_singularity` take the tangent as
//! well as x (the fold test function *is* the tangent's parameter component,
//! and the branch-point one is a determinant bordered by it), and
//! `active_parameter_index` names the component of x carrying the parameter,
//! which the engine needs for the requested direction and the bounds.

use std::cell::Cell;

use crate::linalg::{Lu, null_vector};
use crate::matrix::{Matrix, dot, norm2, norm_inf};
use crate::newton::{NewtonOptions, NewtonSolver, NonlinearSystem};
use crate::types::*;

// ---- Options (§4.3) -------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct ContinuationOptions {
    /// Step sizes, in arclength.
    pub initial_step: f64,
    pub min_step: f64,
    pub max_step: f64,
    pub step_increase_factor: f64,
    pub step_decrease_factor: f64,
    /// 4 rather than 3: finite-difference Jacobians cost Newton its quadratic
    /// rate but not its accuracy (§3.1).
    pub target_newton_iterations: usize,

    pub max_newton_iterations: usize,
    pub tol_residual: f64,
    pub tol_step: f64,
    pub use_line_search: bool,

    pub max_points: usize,
    pub parameter_min: f64,
    pub parameter_max: f64,
    /// Stop if any component of x exceeds this in magnitude (a blow-up).
    pub state_bound_max: f64,

    pub detect_bifurcations: bool,
    pub bisection_tolerance: f64,
    pub max_bisections: usize,
    pub eigenvalue_zero_tol: f64,
    /// Compute normal-form coefficients at each located bifurcation (§6.4).
    /// Carried for the `bifurcata/1` options record; the coefficients
    /// themselves are milestone M3.
    pub compute_normal_forms: bool,

    /// The angle criterion of §5.1: reject a step whose tangent turns by more
    /// than this, which catches a corrector that has landed on a neighbouring
    /// branch — something the iteration count alone will not reveal.
    pub max_tangent_angle_deg: f64,

    pub predictor: PredictorKind,
    pub corrector: CorrectorKind,
}

impl Default for ContinuationOptions {
    fn default() -> Self {
        Self {
            initial_step: 0.01,
            min_step: 1e-6,
            max_step: 0.1,
            step_increase_factor: 1.3,
            step_decrease_factor: 0.5,
            target_newton_iterations: 4,
            max_newton_iterations: 10,
            tol_residual: 1e-10,
            tol_step: 1e-10,
            use_line_search: true,
            max_points: 1000,
            parameter_min: -1e300,
            parameter_max: 1e300,
            state_bound_max: 1e12,
            detect_bifurcations: true,
            bisection_tolerance: 1e-8,
            max_bisections: 40,
            eigenvalue_zero_tol: 1e-8,
            compute_normal_forms: true,
            max_tangent_angle_deg: 30.0,
            predictor: PredictorKind::Tangent,
            corrector: CorrectorKind::Palc,
        }
    }
}

// ---- The curve abstraction (§4.1) -------------------------------------------------

/// A continuable object, as a defining system `G(x) = 0` with `G: ℝᴺ → ℝᴺ⁻¹`,
/// so that its solution set is generically a curve. The engine appends the
/// arclength condition.
pub trait DefiningSystem {
    /// N = dim(x).
    fn system_dim(&self) -> usize;

    /// The index within x of the primary continuation parameter.
    fn active_parameter_index(&self) -> usize;

    /// `G(x)`, of length N − 1.
    fn residual(&self, x: &[f64], g: &mut [f64]);

    /// `DG(x)`, resized to (N − 1) × N.
    fn jacobian(&self, x: &[f64], dg: &mut Matrix);

    fn test_function_count(&self) -> usize;

    /// The test functions at x with tangent `tau`; `psi` has
    /// [`test_function_count`](DefiningSystem::test_function_count) entries.
    fn test_functions(&self, x: &[f64], tau: &[f64], psi: &mut [f64]);

    fn test_function_kind(&self, i: usize) -> BifurcationKind;

    /// Called once a zero of test function `i` has been located. `Err` declines
    /// the candidate, with the reason — normal and often right (a cycle
    /// shrinking back onto its equilibrium, a tangent component passing through
    /// zero without the Jacobian becoming singular).
    fn process_singularity(&self, i: usize, x: &[f64], tau: &[f64]) -> Result<BifurcationInfo, String>;

    /// Unpack x for reporting.
    fn decode(&self, x: &[f64]) -> CurvePoint;
}

impl<T: DefiningSystem + ?Sized> DefiningSystem for &T {
    fn system_dim(&self) -> usize {
        (**self).system_dim()
    }
    fn active_parameter_index(&self) -> usize {
        (**self).active_parameter_index()
    }
    fn residual(&self, x: &[f64], g: &mut [f64]) {
        (**self).residual(x, g)
    }
    fn jacobian(&self, x: &[f64], dg: &mut Matrix) {
        (**self).jacobian(x, dg)
    }
    fn test_function_count(&self) -> usize {
        (**self).test_function_count()
    }
    fn test_functions(&self, x: &[f64], tau: &[f64], psi: &mut [f64]) {
        (**self).test_functions(x, tau, psi)
    }
    fn test_function_kind(&self, i: usize) -> BifurcationKind {
        (**self).test_function_kind(i)
    }
    fn process_singularity(&self, i: usize, x: &[f64], tau: &[f64]) -> Result<BifurcationInfo, String> {
        (**self).process_singularity(i, x, tau)
    }
    fn decode(&self, x: &[f64]) -> CurvePoint {
        (**self).decode(x)
    }
}

// ---- The augmented system -----------------------------------------------------------

/// The system the corrector actually solves (§5.1):
///
/// ```text
/// G(x) = 0
/// τᵀ(x − x_prev) − Δs = 0
/// ```
///
/// expressed as a [`NonlinearSystem`] so that it goes through the same Newton
/// as everything else (§2.1).
pub struct PalcSystem<'a> {
    pub system: &'a dyn DefiningSystem,
    pub x_prev: &'a [f64],
    pub tau: &'a [f64],
    pub ds: f64,
}

impl NonlinearSystem for PalcSystem<'_> {
    fn dim(&self) -> usize {
        self.system.system_dim()
    }

    fn residual(&self, x: &[f64], r: &mut [f64]) {
        let n = self.dim();
        self.system.residual(x, &mut r[..n - 1]);
        r[n - 1] = (0..n).map(|i| self.tau[i] * (x[i] - self.x_prev[i])).sum::<f64>() - self.ds;
    }

    fn jacobian(&self, x: &[f64], j: &mut Matrix) {
        *j = bordered_jacobian(self.system, x, self.tau);
    }
}

/// `[DG(x); borderᵀ]`, N × N. The bordering row keeps it nonsingular at a
/// fold — the whole reason pseudo-arclength is used rather than natural
/// parameter continuation (§5.1).
fn bordered_jacobian(system: &dyn DefiningSystem, x: &[f64], border: &[f64]) -> Matrix {
    let n = system.system_dim();
    let mut dg = Matrix::zeros(n - 1, n);
    system.jacobian(x, &mut dg);
    Matrix::from_fn(n, n, |i, j| if i < n - 1 { dg[(i, j)] } else { border[j] })
}

// ---- Events --------------------------------------------------------------------------

/// One corrector attempt, rejected ones included: §10.4's trace wants exactly
/// this, because a baseline mismatch is usually explained by the step-size
/// history in the few points before it was noticed.
#[derive(Clone, Debug, PartialEq)]
pub struct Attempt {
    pub point_index: usize,
    pub attempt: usize,
    pub step_size: f64,
    pub iterations: usize,
    pub residual_norm: f64,
    pub accepted: bool,
    pub tangent_angle_deg: f64,
    pub detail: String,
}

/// A test function changed sign, its zero was located, and the defining system
/// declined to call it a bifurcation. Without this, a genuine detection lost to
/// an over-strict guard is indistinguishable from a test function that never
/// fired at all.
#[derive(Clone, Debug, PartialEq)]
pub struct Declined {
    pub index: usize,
    /// The kind the test function would have reported.
    pub kind: BifurcationKind,
    /// The active parameter at the located point.
    pub parameter: f64,
    pub detail: String,
}

type PointCallback = Box<dyn FnMut(&CurvePoint) -> bool>;
type AttemptCallback = Box<dyn FnMut(&Attempt)>;
type DeclinedCallback = Box<dyn FnMut(&Declined)>;

// ---- The engine (§4.2) -----------------------------------------------------------------

pub struct ContinuationEngine<S: DefiningSystem> {
    system: S,
    options: ContinuationOptions,
    newton: NewtonSolver,
    branch: Branch,
    n: usize,

    x: Vec<f64>,
    tau: Vec<f64>,
    secant: Option<Vec<f64>>,
    ds: f64,
    s: f64,
    psi: Vec<f64>,

    initialised: bool,
    terminated: bool,
    last_status: SolveStatus,
    next_bifurcation_id: usize,
    last_predictor: Cell<PredictorKind>,
    last_corrector: Cell<CorrectorKind>,

    on_point: Option<PointCallback>,
    on_attempt: Option<AttemptCallback>,
    on_declined: Option<DeclinedCallback>,
}

/// The sign as −1, 0 or +1 (Delphi's `Sign`, which `f64::signum` is not: it
/// gives ±1 for ±0).
fn sign(v: f64) -> i32 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

fn normalise(v: &mut [f64]) -> f64 {
    let norm = norm2(v);
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    norm
}

impl<S: DefiningSystem> ContinuationEngine<S> {
    /// An engine for `system`, recording into a new branch with id 0.
    pub fn new(system: S, options: ContinuationOptions) -> Self {
        Self::with_branch(system, options, Branch::new(0))
    }

    /// An engine recording into `branch` (whose id, parent and origin the
    /// caller has set).
    pub fn with_branch(system: S, options: ContinuationOptions, branch: Branch) -> Self {
        let n = system.system_dim();
        let newton = NewtonSolver::new(NewtonOptions {
            max_iterations: options.max_newton_iterations,
            tol_residual: options.tol_residual,
            tol_step: options.tol_step,
            use_line_search: options.use_line_search,
            ..NewtonOptions::default()
        });
        let ds = options.initial_step;
        Self {
            system,
            options,
            newton,
            branch,
            n,
            x: Vec::new(),
            tau: Vec::new(),
            secant: None,
            ds,
            s: 0.0,
            psi: Vec::new(),
            initialised: false,
            terminated: false,
            last_status: SolveStatus::ok(),
            next_bifurcation_id: 0,
            last_predictor: Cell::new(PredictorKind::Tangent),
            last_corrector: Cell::new(CorrectorKind::Palc),
            on_point: None,
            on_attempt: None,
            on_declined: None,
        }
    }

    pub fn system(&self) -> &S {
        &self.system
    }

    pub fn options(&self) -> &ContinuationOptions {
        &self.options
    }

    pub fn branch(&self) -> &Branch {
        &self.branch
    }

    pub fn into_branch(self) -> Branch {
        self.branch
    }

    pub fn last_status(&self) -> &SolveStatus {
        &self.last_status
    }

    pub fn current_step_size(&self) -> f64 {
        self.ds
    }

    pub fn arclength(&self) -> f64 {
        self.s
    }

    /// The full unknown vector at the current point.
    pub fn current_x(&self) -> &[f64] {
        &self.x
    }

    /// The unit tangent at the current point.
    pub fn current_tangent(&self) -> &[f64] {
        &self.tau
    }

    /// Which predictor actually ran on the last step: a secant predictor has no
    /// history on the first step and falls back to the tangent.
    pub fn last_predictor_used(&self) -> PredictorKind {
        self.last_predictor.get()
    }

    pub fn last_corrector_used(&self) -> CorrectorKind {
        self.last_corrector.get()
    }

    /// Bifurcation ids are allocated per engine; a caller building a tree of
    /// branches seeds this with a running total to keep them unique.
    pub fn next_bifurcation_id(&self) -> usize {
        self.next_bifurcation_id
    }

    pub fn set_next_bifurcation_id(&mut self, id: usize) {
        self.next_bifurcation_id = id;
    }

    /// Called with every recorded point; returning true aborts the run.
    pub fn on_point(&mut self, f: impl FnMut(&CurvePoint) -> bool + 'static) {
        self.on_point = Some(Box::new(f));
    }

    pub fn on_attempt(&mut self, f: impl FnMut(&Attempt) + 'static) {
        self.on_attempt = Some(Box::new(f));
    }

    pub fn on_declined(&mut self, f: impl FnMut(&Declined) + 'static) {
        self.on_declined = Some(Box::new(f));
    }

    fn sys(&self) -> &dyn DefiningSystem {
        &self.system
    }

    // ---- Tangents ----

    /// Solve `[DG(x); τ_refᵀ] τ = e_N` and normalise. Bordering with the
    /// previous tangent both fixes the scale and selects the direction of
    /// travel, which prevents the reversal that plagues naive implementations
    /// at folds (§5.1).
    fn compute_tangent(&self, x: &[f64], tau_ref: &[f64]) -> Result<Vec<f64>, SolveStatus> {
        let lu = Lu::new(&bordered_jacobian(self.sys(), x, tau_ref))?;
        let mut e = vec![0.0; self.n];
        e[self.n - 1] = 1.0;
        let mut tau = lu.solve(&e)?;
        if normalise(&mut tau) == 0.0 {
            return Err(SolveStatus::fail(SolveCode::SingularJacobian, "the tangent computation produced a zero vector"));
        }
        Ok(tau)
    }

    /// With no previous tangent to border with: the null vector of the
    /// (N − 1) × N Jacobian (§6.1), oriented by the requested direction.
    fn initial_tangent(&self, x: &[f64], direction: i32) -> Result<Vec<f64>, SolveStatus> {
        let mut dg = Matrix::zeros(self.n - 1, self.n);
        self.system.jacobian(x, &mut dg);
        let (mut tau, _) = null_vector(&dg).map_err(|e| SolveStatus::fail(SolveCode::LinAlgError, format!("initial tangent: {}", e.detail)))?;
        let k = self.system.active_parameter_index();
        // If the parameter component is numerically zero the start is already
        // at a fold and the direction is ambiguous: orient by the largest
        // component, so at least the choice is deterministic.
        let k = if tau[k].abs() > 1e-12 {
            k
        } else {
            let largest = norm_inf(&tau);
            tau.iter().position(|v| v.abs() == largest).unwrap_or(k)
        };
        if sign(tau[k]) != sign(direction as f64) {
            tau.iter_mut().for_each(|v| *v = -*v);
        }
        Ok(tau)
    }

    // ---- Prediction and correction ----

    /// The predictor supplies only the initial guess; both kinds hand it to the
    /// same corrector, whose border is always the tangent (§4.3 lists them as
    /// independent settings).
    fn predict(&self, x_prev: &[f64], tau: &[f64], ds: f64) -> Vec<f64> {
        let direction = match (&self.options.predictor, &self.secant) {
            (PredictorKind::Secant, Some(secant)) => {
                self.last_predictor.set(PredictorKind::Secant);
                secant
            }
            _ => {
                self.last_predictor.set(PredictorKind::Tangent);
                tau
            }
        };
        x_prev.iter().zip(direction).map(|(x, d)| x + ds * d).collect()
    }

    /// The PALC corrector: Newton on the augmented system. Returns the point
    /// and the iterations it took.
    fn correct_palc(&self, x_prev: &[f64], tau: &[f64], ds: f64) -> Result<(Vec<f64>, usize), SolveStatus> {
        let mut x = self.predict(x_prev, tau, ds);
        let palc = PalcSystem { system: self.sys(), x_prev, tau, ds };
        let report = self.newton.solve(&palc, &mut x)?;
        Ok((x, report.iterations))
    }

    /// The Moore–Penrose corrector (§5.2): each iteration borders with the
    /// current iterate's tangent and moves orthogonally to it — a Gauss–Newton
    /// step on the underdetermined G(x) = 0, converging to the nearest point on
    /// the curve rather than to a prescribed arclength (Allgower & Georg;
    /// MatCont's default).
    ///
    /// Deliberately not routed through [`NewtonSolver`]: §2.1 objects to two
    /// Newtons solving the same problem, and this one solves a different one,
    /// with a border that moves between iterations. It shares the tolerances
    /// and iteration budget so the two correctors stay comparable.
    fn correct_moore_penrose(&self, x_prev: &[f64], tau: &[f64], ds: f64) -> Result<(Vec<f64>, Vec<f64>, usize), SolveStatus> {
        let n = self.n;
        let o = &self.options;
        let mut x = self.predict(x_prev, tau, ds);
        let mut v = tau.to_vec();
        let mut g = vec![0.0; n - 1];
        let mut iterations = 0;
        let residual = |x: &[f64], g: &mut [f64]| {
            self.system.residual(x, g);
            norm_inf(g)
        };

        for k in 1..=o.max_newton_iterations {
            let residual_norm = residual(&x, &mut g);
            if residual_norm <= o.tol_residual {
                return Ok((x, v, iterations));
            }
            let lu = Lu::new(&bordered_jacobian(self.sys(), &x, &v)).map_err(|e| SolveStatus {
                detail: format!("Moore–Penrose: the bordered system is singular at iteration {k} ({})", e.detail),
                iterations,
                residual_norm,
                ..e
            })?;
            // [DG; vᵀ] δ = [−G; 0]: the zero makes the correction orthogonal to v.
            let mut rhs: Vec<f64> = g.iter().map(|v| -v).collect();
            rhs.push(0.0);
            let delta = lu.solve(&rhs)?;
            let step_norm = norm_inf(&delta);
            x.iter_mut().zip(&delta).for_each(|(x, d)| *x += d);
            iterations += 1;

            // Refresh the tangent from the same factorisation, keeping its orientation.
            let mut e = vec![0.0; n];
            e[n - 1] = 1.0;
            if let Ok(mut t) = lu.solve(&e)
                && normalise(&mut t) > 0.0
            {
                if dot(&t, &v) < 0.0 {
                    t.iter_mut().for_each(|c| *c = -*c);
                }
                v = t;
            }

            if step_norm <= o.tol_step {
                let residual_norm = residual(&x, &mut g);
                if residual_norm <= o.tol_residual.sqrt() {
                    return Ok((x, v, iterations));
                }
                return Err(SolveStatus {
                    code: SolveCode::NotConverged,
                    detail: format!("Moore–Penrose: the step collapsed to {step_norm:.3e} but the residual is still {residual_norm:.3e}"),
                    iterations,
                    residual_norm,
                });
            }
        }
        let residual_norm = residual(&x, &mut g);
        Err(SolveStatus {
            code: SolveCode::MaxIterations,
            detail: format!(
                "Moore–Penrose: {} iterations exhausted, residual {residual_norm:.3e} (tolerance {:.3e})",
                o.max_newton_iterations, o.tol_residual
            ),
            iterations: o.max_newton_iterations,
            residual_norm,
        })
    }

    /// One entry point for both correctors: the new point, its tangent, and the
    /// iterations taken.
    fn correct_and_tangent(&self, x_prev: &[f64], tau: &[f64], ds: f64) -> Result<(Vec<f64>, Vec<f64>, usize), SolveStatus> {
        match self.options.corrector {
            CorrectorKind::MoorePenrose => {
                self.last_corrector.set(CorrectorKind::MoorePenrose);
                self.correct_moore_penrose(x_prev, tau, ds)
            }
            CorrectorKind::Palc => {
                self.last_corrector.set(CorrectorKind::Palc);
                let (x, iterations) = self.correct_palc(x_prev, tau, ds)?;
                let tau_new = self.compute_tangent(&x, tau).map_err(|e| SolveStatus { iterations, ..e })?;
                Ok((x, tau_new, iterations))
            }
        }
    }

    // ---- Scaling ----

    /// Scale the corrector by the row norms of DG (§6.1). Not by the residual:
    /// the starting point is on the curve and G is already zero there. Row
    /// equilibration makes each equation's sensitivity comparable, which is
    /// what the convergence test needs — without it a model whose rate laws
    /// differ by orders of magnitude is judged by its largest one alone, and
    /// driving that to an absolute 1e-10 can demand more digits than the
    /// arithmetic has.
    fn set_typical_from_jacobian(&mut self, x: &[f64]) {
        let n = self.n;
        let mut dg = Matrix::zeros(n - 1, n);
        self.system.jacobian(x, &mut dg);
        let mut typical: Vec<f64> = (0..n - 1).map(|row| (0..n).fold(0.0f64, |m, col| m.max(dg[(row, col)].abs()))).collect();
        let biggest = typical.iter().fold(0.0f64, |m, v| m.max(*v));
        if biggest <= 0.0 {
            return; // a zero Jacobian says nothing; leave the defaults
        }
        // A numerically empty row must not become a near-zero divisor.
        typical.iter_mut().for_each(|v| *v = v.max(biggest * 1e-10));
        // The arclength row is already O(1) in the units of the step; scaling it
        // by a Jacobian row norm would compare an arclength with a reaction rate.
        typical.push(1.0);
        self.newton.set_typical_f(typical);
    }

    // ---- Initialise and step ----

    /// Correct `x0` onto the curve, compute the initial tangent and record the
    /// first point. `direction` is +1 or −1: the sign of the tangent's
    /// parameter component.
    pub fn initialise(&mut self, x0: &[f64], direction: i32) -> Result<(), SolveStatus> {
        if x0.len() != self.n {
            let status = SolveStatus::fail(SolveCode::InvalidInput, format!("initialise: x0 has length {}, the system dimension is {}", x0.len(), self.n));
            self.last_status = status.clone();
            return Err(status);
        }
        let mut x = x0.to_vec();
        // Correct onto the curve with the arclength row disabled (a zero border
        // and step): the augmented matrix is then singular, and the
        // Levenberg–Marquardt fallback solves G = 0 in the least-squares sense.
        // In practice x0 comes from a steady-state solve and is already on the
        // curve; this is a safety net, and its outcome is judged by the tangent.
        let zero = vec![0.0; self.n];
        let _ = self.newton.solve(&PalcSystem { system: self.sys(), x_prev: x0, tau: &zero, ds: 0.0 }, &mut x);

        let tau = match self.initial_tangent(&x, direction) {
            Ok(t) => t,
            Err(e) => {
                self.last_status = e.clone();
                return Err(e);
            }
        };
        self.set_typical_from_jacobian(&x);

        self.x = x;
        self.tau = tau;
        self.s = 0.0;
        self.ds = self.options.initial_step;
        // No previous point: a secant predictor falls back to the tangent.
        self.secant = None;
        self.psi = vec![0.0; self.system.test_function_count()];
        if self.options.detect_bifurcations {
            self.system.test_functions(&self.x, &self.tau, &mut self.psi);
        }
        self.initialised = true;
        self.terminated = false;
        self.last_status = SolveStatus::ok();
        // An abort requested on the very first point takes effect at the first step.
        if self.record_point(0.0, 0) {
            self.terminate(TerminationReason::UserAborted, "");
        }
        Ok(())
    }

    fn terminate(&mut self, reason: TerminationReason, detail: &str) {
        self.branch.termination = reason;
        self.branch.termination_detail = detail.to_owned();
        self.terminated = true;
    }

    /// Advance by one point.
    pub fn step(&mut self) -> StepResult {
        if !self.initialised {
            self.last_status = SolveStatus::fail(SolveCode::InvalidInput, "step: the engine has not been initialised");
            return StepResult::Failed;
        }
        if self.terminated {
            return StepResult::Converged;
        }
        if self.branch.points.len() >= self.options.max_points {
            self.terminate(TerminationReason::MaxPoints, "");
            return StepResult::Converged;
        }

        let min_cos = self.options.max_tangent_angle_deg.to_radians().cos();
        let mut attempt = 0;
        let (x_new, tau_new, iterations) = loop {
            attempt += 1;
            let mut angle = 0.0;
            let outcome = self.correct_and_tangent(&self.x, &self.tau, self.ds).and_then(|(x, t, it)| {
                // A corrector that converged onto a DIFFERENT branch usually does
                // so in a normal number of iterations; only the tangent turning
                // reveals it.
                let cos = dot(&self.tau, &t);
                angle = cos.clamp(-1.0, 1.0).acos().to_degrees();
                if cos >= min_cos {
                    Ok((x, t, it))
                } else {
                    Err(SolveStatus {
                        code: SolveCode::NotConverged,
                        detail: format!("the tangent turned by {angle:.1} degrees, limit {:.1}", self.options.max_tangent_angle_deg),
                        iterations: it,
                        residual_norm: 0.0,
                    })
                }
            });
            if let Some(f) = self.on_attempt.as_mut() {
                let (iterations, residual_norm, detail) = match &outcome {
                    Ok((_, _, it)) => (*it, 0.0, String::new()),
                    Err(e) => (e.iterations, e.residual_norm, e.detail.clone()),
                };
                f(&Attempt {
                    point_index: self.branch.points.len(),
                    attempt,
                    step_size: self.ds,
                    iterations,
                    residual_norm,
                    accepted: outcome.is_ok(),
                    tangent_angle_deg: angle,
                    detail,
                });
            }
            match outcome {
                Ok(accepted) => break accepted,
                Err(status) => {
                    // Halve and retry from the same point (§5.1).
                    self.ds *= self.options.step_decrease_factor;
                    if self.ds.abs() < self.options.min_step {
                        self.terminate(TerminationReason::StepTooSmall, &status.detail);
                        self.last_status = status;
                        return StepResult::Failed;
                    }
                }
            }
        };

        // Accepted.
        let x_prev = std::mem::replace(&mut self.x, x_new);
        let tau_prev = std::mem::replace(&mut self.tau, tau_new);
        let mut secant: Vec<f64> = self.x.iter().zip(&x_prev).map(|(a, b)| a - b).collect();
        self.secant = (normalise(&mut secant) > 0.0).then_some(secant);
        let s_prev = self.s;
        self.s += self.ds.abs();

        let psi_prev = self.psi.clone();
        if self.options.detect_bifurcations {
            self.system.test_functions(&self.x, &self.tau, &mut self.psi);
        }
        let abort = self.record_point(self.ds, iterations);
        if self.options.detect_bifurcations {
            let psi = self.psi.clone();
            self.detect_and_locate(&x_prev, &tau_prev, &psi_prev, &psi, self.ds, s_prev);
        }

        if abort {
            self.terminate(TerminationReason::UserAborted, "");
            return StepResult::Converged;
        }
        if let Some(reason) = self.out_of_bounds() {
            self.terminate(reason, "");
            return StepResult::Boundary;
        }

        // Step adaptation (§5.1).
        if iterations <= self.options.target_newton_iterations {
            self.ds = (self.ds.abs() * self.options.step_increase_factor).min(self.options.max_step) * self.ds.signum();
        }
        StepResult::Ok
    }

    /// Step until the branch ends; returns the last result.
    pub fn run(&mut self) -> StepResult {
        loop {
            let result = self.step();
            if result != StepResult::Ok {
                return result;
            }
        }
    }

    fn out_of_bounds(&self) -> Option<TerminationReason> {
        let p = self.x[self.system.active_parameter_index()];
        if p < self.options.parameter_min || p > self.options.parameter_max {
            return Some(TerminationReason::ParameterBound);
        }
        if self.x.iter().any(|v| v.is_nan() || v.abs() > self.options.state_bound_max) {
            return Some(TerminationReason::StateBound);
        }
        None
    }

    /// Record the current point; true if the callback asked to abort.
    fn record_point(&mut self, ds: f64, iterations: usize) -> bool {
        let mut p = self.system.decode(&self.x);
        p.s = self.s;
        p.step_size = ds;
        p.newton_iterations = iterations;
        if self.options.detect_bifurcations {
            p.test_functions = self.psi.clone();
        }
        let abort = self.on_point.as_mut().is_some_and(|f| f(&p));
        self.branch.add_point(p);
        abort
    }

    // ---- Detection and location (§6.3) ----

    fn detect_and_locate(&mut self, x_a: &[f64], tau_a: &[f64], psi_a: &[f64], psi_b: &[f64], ds: f64, s_a: f64) {
        for i in 0..psi_a.len() {
            // Exact zeros at an endpoint are skipped deliberately: they recur at
            // the next step and would be reported twice.
            if psi_a[i] == 0.0 || psi_b[i] == 0.0 {
                continue;
            }
            if sign(psi_a[i]) != sign(psi_b[i]) {
                self.refine_and_locate(i, x_a, tau_a, psi_a[i], psi_b[i], ds, s_a);
            }
        }
    }

    /// Walk the bracket again in short genuine continuation steps before
    /// bisecting.
    ///
    /// Bisection predicts from one end along a fixed tangent and corrects. Near
    /// a branch point the corrector fails on exactly the trials closest to the
    /// zero, so bisection falls back to the nearest trial that did correct —
    /// and how near that is scales with the bracket it started from. Every test
    /// function is built on the tangent, too, so a zero found at the maximum
    /// step inherits that step's error. Each level here shortens the bracket
    /// eightfold with real steps, each with its own tangent: on MatCont's Lab 2
    /// this took the branch point from 1.2% out to nine figures.
    #[allow(clippy::too_many_arguments)]
    fn refine_and_locate(&mut self, index: usize, x_a: &[f64], tau_a: &[f64], psi_lo: f64, mut psi_hi: f64, ds_hi: f64, s_a: f64) {
        const SUBDIVISIONS: usize = 8;
        const MAX_LEVELS: usize = 3;
        let mut anchor = x_a.to_vec();
        let mut anchor_tau = tau_a.to_vec();
        let mut anchor_s = s_a;
        let mut psi_anchor = psi_lo;
        let mut length = ds_hi;
        let mut psi = vec![0.0; self.system.test_function_count()];

        for _ in 0..MAX_LEVELS {
            // Below the corrector's own step tolerance there is nothing to gain.
            if length.abs() <= 8.0 * self.options.min_step {
                break;
            }
            let sub = length / SUBDIVISIONS as f64;
            let mut found = false;
            for _ in 0..SUBDIVISIONS {
                // A corrector failure inside the bracket is itself the sign that
                // the zero is near: stop and bisect what has been narrowed.
                let Some((x, tau)) = self.correct_on_branch(&anchor, &anchor_tau, sub) else {
                    break;
                };
                self.system.test_functions(&x, &tau, &mut psi);
                let next = psi[index];
                if next != 0.0 && psi_anchor != 0.0 && sign(next) != sign(psi_anchor) {
                    psi_hi = next;
                    length = sub;
                    found = true;
                    break;
                }
                anchor = x;
                anchor_tau = tau;
                anchor_s += sub.abs();
                psi_anchor = next;
            }
            if !found {
                break;
            }
        }
        self.locate_by_bisection(index, &anchor, &anchor_tau, psi_anchor, psi_hi, length, anchor_s);
    }

    /// A PALC correction and its tangent for a location trial, or `None` if the
    /// corrector fails or the tangent turns by more than the step's angle limit.
    ///
    /// The angle test is the one each step already applies (§5.1), and a
    /// location needs it more: near a branch point both crossing branches meet
    /// the arclength hyperplane, and a trial that converges onto the other one
    /// sits at a small |ψ| and wins the bisection. A safeguard the Delphi
    /// version lacks: on PP2's fish-only branch, where it crosses the trivial
    /// one at p1 = 0.6, a corrector without its row scaling did exactly this
    /// and reported the branch point at 0.5961 — a point on the trivial branch.
    fn correct_on_branch(&self, x_a: &[f64], tau_a: &[f64], ds: f64) -> Option<(Vec<f64>, Vec<f64>)> {
        let (x, _) = self.correct_palc(x_a, tau_a, ds).ok()?;
        let tau = self.compute_tangent(&x, tau_a).ok()?;
        (dot(&tau, tau_a) >= self.options.max_tangent_angle_deg.to_radians().cos()).then_some((x, tau))
    }

    #[allow(clippy::too_many_arguments)]
    fn locate_by_bisection(&mut self, index: usize, x_a: &[f64], tau_a: &[f64], mut psi_lo: f64, _psi_hi: f64, ds_hi: f64, s_a: f64) {
        let (mut lo, mut hi) = (0.0, ds_hi);
        let mut psi = vec![0.0; self.system.test_function_count()];
        // Track the best point actually reached rather than trusting the last
        // midpoint: near a branch point the corrector fails on exactly the
        // trials closest to the zero.
        let mut best_x = x_a.to_vec();
        let mut best_tau = tau_a.to_vec();
        let mut best_abs = psi_lo.abs();
        let mut best_s = 0.0;
        let mut have_best = false;

        for _ in 0..self.options.max_bisections {
            let mid = 0.5 * (lo + hi);
            let Some((x_mid, tau_mid)) = self.correct_on_branch(x_a, tau_a, mid) else {
                // Cannot correct here, or landed on another branch: narrow
                // towards the side we came from.
                hi = mid;
                continue;
            };
            self.system.test_functions(&x_mid, &tau_mid, &mut psi);
            let at_mid = psi[index];
            if !have_best || at_mid.abs() < best_abs {
                best_x = x_mid;
                best_tau = tau_mid;
                best_abs = at_mid.abs();
                best_s = mid;
                have_best = true;
            }
            if at_mid.abs() < self.options.bisection_tolerance {
                break;
            }
            if sign(at_mid) == sign(psi_lo) {
                lo = mid;
                psi_lo = at_mid;
            } else {
                hi = mid;
            }
            if hi - lo < self.options.bisection_tolerance * ds_hi.max(1.0) {
                break;
            }
        }

        match self.system.process_singularity(index, &best_x, &best_tau) {
            Ok(mut info) => {
                // The tangent we arrived on: at a branch point two curves cross,
                // and which one to switch onto is defined by which one we came
                // in along. Only the engine knows it.
                info.point.tangent = best_tau;
                info.point.raw_x = best_x;
                info.id = self.next_bifurcation_id;
                self.next_bifurcation_id += 1;
                info.branch_id = self.branch.id;
                info.point.s = s_a + best_s.abs();
                // Test functions are built on the tangent, so a point located
                // at the maximum step is only as accurate as that step.
                info.point.step_size = ds_hi.abs();
                if ds_hi.abs() >= 0.99 * self.options.max_step {
                    info.detail.push_str(&format!(
                        "  [located while stepping at the maximum step size ({:.3}), so its position is only as accurate as that step; \
                         re-run with a smaller maxStep to sharpen it]",
                        ds_hi.abs()
                    ));
                }
                self.branch.add_bifurcation(info);
            }
            Err(detail) => {
                if let Some(f) = self.on_declined.as_mut() {
                    f(&Declined {
                        index,
                        kind: self.system.test_function_kind(index),
                        parameter: best_x[self.system.active_parameter_index()],
                        detail,
                    });
                }
            }
        }
    }
}
