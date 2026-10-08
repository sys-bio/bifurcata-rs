//! Hand-written problems with known answers, so the library can be tested
//! with no model backend at all, and against mathematics rather than against a
//! previous run of itself. Ported from the Delphi `Bifurcata.TestProblems`.

use std::f64::consts::PI;

use crate::matrix::Matrix;
use crate::problem::BifurcationProblem;

/// The Brusselator (§11.2):
///
/// ```text
/// dX/dt = A − (B+1) X + X² Y
/// dY/dt = B X − X² Y
/// ```
///
/// * equilibrium X* = A, Y* = B/A (unique for A ≠ 0)
/// * Jacobian there [[B−1, A²], [−B, −A²]]: trace B − 1 − A², determinant A²
/// * Hopf at B = 1 + A² with ω = A
/// * first Lyapunov coefficient l1 = −1/2 at A = 1, B = 2 (Kuznetsov's
///   normalisation; the −0.0625 once printed in the spec is wrong)
///
/// The determinant A² > 0 everywhere means there is **no fold**: a fold test
/// function that fires here is wrong. Parameters are [A, B], so B — the usual
/// continuation parameter — is index 1.
pub struct Brusselator;

impl Brusselator {
    pub fn equilibrium(a: f64, b: f64) -> [f64; 2] {
        [a, b / a]
    }

    pub fn hopf_b(a: f64) -> f64 {
        1.0 + a * a
    }

    pub fn hopf_omega(a: f64) -> f64 {
        a.abs()
    }
}

impl BifurcationProblem for Brusselator {
    fn state_dim(&self) -> usize {
        2
    }
    fn parameter_count(&self) -> usize {
        2
    }
    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
        let (x, y, a, b) = (u[0], u[1], lambda[0], lambda[1]);
        r[0] = a - (b + 1.0) * x + x * x * y;
        r[1] = b * x - x * x * y;
    }
    fn jacobian(&self, u: &[f64], lambda: &[f64], j: &mut Matrix) {
        let (x, y, b) = (u[0], u[1], lambda[1]);
        j.resize_zeroed(2, 2);
        j[(0, 0)] = -(b + 1.0) + 2.0 * x * y;
        j[(0, 1)] = x * x;
        j[(1, 0)] = b - 2.0 * x * y;
        j[(1, 1)] = -x * x;
    }
    fn parameter_derivative(&self, u: &[f64], _lambda: &[f64], k: usize, out: &mut [f64]) {
        match k {
            0 => out.copy_from_slice(&[1.0, 0.0]),
            1 => out.copy_from_slice(&[-u[0], u[0]]),
            _ => panic!("Brusselator has no parameter {k}"),
        }
    }
    fn state_name(&self, i: usize) -> String {
        ["X", "Y"][i].to_owned()
    }
    fn parameter_name(&self, k: usize) -> String {
        ["A", "B"][k].to_owned()
    }
}

/// A saddle-node in one state: `du/dt = λ − u²`. Equilibria u = ±√λ for
/// λ > 0 meet at a fold at λ = 0, where the Jacobian −2u is exactly singular —
/// the case that breaks natural-parameter continuation and that the bordered
/// corrector has to survive.
pub struct FoldProblem;

impl BifurcationProblem for FoldProblem {
    fn state_dim(&self) -> usize {
        1
    }
    fn parameter_count(&self) -> usize {
        1
    }
    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
        r[0] = lambda[0] - u[0] * u[0];
    }
    fn jacobian(&self, u: &[f64], _lambda: &[f64], j: &mut Matrix) {
        j.resize_zeroed(1, 1);
        j[(0, 0)] = -2.0 * u[0];
    }
    fn state_name(&self, _i: usize) -> String {
        "u".to_owned()
    }
    fn parameter_name(&self, _k: usize) -> String {
        "lambda".to_owned()
    }
}

/// A badly scaled linear system, for the scaling requirement of §6.1:
/// `dU/dt = s_u (U − U*)`, `dV/dt = s_v (V − V*)`. With s_u ≈ 1e6 and
/// s_v ≈ 1e-6 the residuals differ by twelve orders of magnitude, so an
/// unscaled convergence test is decided by the first equation alone.
pub struct BadlyScaledProblem {
    pub scale_u: f64,
    pub scale_v: f64,
    pub target_u: f64,
    pub target_v: f64,
}

impl BifurcationProblem for BadlyScaledProblem {
    fn state_dim(&self) -> usize {
        2
    }
    fn parameter_count(&self) -> usize {
        1
    }
    fn residual(&self, u: &[f64], _lambda: &[f64], r: &mut [f64]) {
        r[0] = self.scale_u * (u[0] - self.target_u);
        r[1] = self.scale_v * (u[1] - self.target_v);
    }
    fn jacobian(&self, _u: &[f64], _lambda: &[f64], j: &mut Matrix) {
        j.resize_zeroed(2, 2);
        j[(0, 0)] = self.scale_u;
        j[(1, 1)] = self.scale_v;
    }
}

/// The n-stage Goodwin oscillator, an arbitrary-dimension test with an
/// analytic Hopf:
///
/// ```text
/// dx₀/dt = 1/(1 + x_{n−1}^p) − x₀
/// dxᵢ/dt = x_{i−1} − xᵢ          (i = 1 … n−1)
/// ```
///
/// The steady state is uniform, x(1 + x^p) = 1. The characteristic equation is
/// (μ + 1)ⁿ + g = 0 with g = p(1 − x), so the Hopf is at g = sec(π/n)ⁿ with
/// ω = tan(π/n) — independent of p and of the steady state, an exact check
/// that costs nothing. For n = 3 this is the classical result that a Hill
/// coefficient giving g ≥ 8 is needed to oscillate.
pub struct GoodwinProblem {
    n: usize,
}

impl GoodwinProblem {
    /// At least 3 stages are needed for a Hopf.
    pub fn new(stages: usize) -> Self {
        assert!(stages >= 3, "GoodwinProblem: {stages} stages; at least 3 are needed for a Hopf");
        Self { n: stages }
    }

    /// g at the Hopf: sec(π/n)ⁿ.
    pub fn critical_g(n: usize) -> f64 {
        (1.0 / (PI / n as f64).cos()).powi(n as i32)
    }

    /// ω at the Hopf: tan(π/n).
    pub fn hopf_omega(n: usize) -> f64 {
        (PI / n as f64).tan()
    }

    /// The uniform steady state for Hill coefficient p: x(1 + x^p) = 1, by bisection.
    pub fn steady_state_value(p: f64) -> f64 {
        let (mut lo, mut hi): (f64, f64) = (1e-12, 1.0);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid * (1.0 + mid.powf(p)) - 1.0 > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// The Hill coefficient at the Hopf, from p(1 − x) = g_crit.
    ///
    /// g grows only like ln p, so small n needs enormous p (n = 3 needs about
    /// 24 000). The bracket is checked rather than assumed: returning the
    /// search bound would be a plausible-looking wrong answer.
    pub fn critical_p(n: usize) -> Result<f64, String> {
        const P_MAX: f64 = 1e7;
        let gc = Self::critical_g(n);
        let x = Self::steady_state_value(P_MAX);
        if P_MAX * (1.0 - x) < gc {
            return Err(format!(
                "n = {n} needs a Hill coefficient beyond {P_MAX:.3e} (g reaches only {:.4} against a threshold of {gc:.4})",
                P_MAX * (1.0 - x)
            ));
        }
        let (mut lo, mut hi) = (1.0, P_MAX);
        for _ in 0..300 {
            let mid = 0.5 * (lo + hi);
            if mid * (1.0 - Self::steady_state_value(mid)) > gc {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Ok(0.5 * (lo + hi))
    }
}

impl BifurcationProblem for GoodwinProblem {
    fn state_dim(&self) -> usize {
        self.n
    }
    fn parameter_count(&self) -> usize {
        1
    }
    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
        let (p, xn) = (lambda[0], u[self.n - 1]);
        let xp = if xn <= 0.0 { 0.0 } else { xn.powf(p) };
        r[0] = 1.0 / (1.0 + xp) - u[0];
        for i in 1..self.n {
            r[i] = u[i - 1] - u[i];
        }
    }
    fn jacobian(&self, u: &[f64], lambda: &[f64], j: &mut Matrix) {
        let n = self.n;
        let (p, xn) = (lambda[0], u[n - 1]);
        j.resize_zeroed(n, n);
        // d/dx [1/(1 + x^p)] = −p x^(p−1) / (1 + x^p)²
        let g = if xn <= 0.0 { 0.0 } else { p * xn.powf(p - 1.0) / (1.0 + xn.powf(p)).powi(2) };
        j[(0, 0)] = -1.0;
        j[(0, n - 1)] -= g;
        for i in 1..n {
            j[(i, i - 1)] = 1.0;
            j[(i, i)] = -1.0;
        }
    }
    fn state_name(&self, i: usize) -> String {
        format!("x{i}")
    }
    fn parameter_name(&self, _k: usize) -> String {
        "p".to_owned()
    }
}

/// A transcritical bifurcation — a simple **branch point**: `du/dt = λu − u²`.
/// The branches u = 0 and u = λ cross at the origin. On u = 0 the
/// tangent-bordered determinant is proportional to λ and changes sign while the
/// tangent's parameter component stays ±1: the branch-point test fires and the
/// fold test does not, the discrimination §6.3 requires.
pub struct TranscriticalProblem;

impl BifurcationProblem for TranscriticalProblem {
    fn state_dim(&self) -> usize {
        1
    }
    fn parameter_count(&self) -> usize {
        1
    }
    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
        r[0] = lambda[0] * u[0] - u[0] * u[0];
    }
    fn jacobian(&self, u: &[f64], lambda: &[f64], j: &mut Matrix) {
        j.resize_zeroed(1, 1);
        j[(0, 0)] = lambda[0] - 2.0 * u[0];
    }
    fn parameter_derivative(&self, u: &[f64], _lambda: &[f64], _k: usize, out: &mut [f64]) {
        out[0] = u[0];
    }
    fn state_name(&self, _i: usize) -> String {
        "u".to_owned()
    }
    fn parameter_name(&self, _k: usize) -> String {
        "lambda".to_owned()
    }
}

/// The textbook line-search case, `du/dt = arctan(u − λ)`. Undamped Newton
/// diverges from any |u − λ| above about 1.39 (each step overshoots further);
/// with Armijo backtracking it converges. Success and failure differ by the
/// line search alone.
pub struct ArctanProblem;

impl BifurcationProblem for ArctanProblem {
    fn state_dim(&self) -> usize {
        1
    }
    fn parameter_count(&self) -> usize {
        1
    }
    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
        r[0] = (u[0] - lambda[0]).atan();
    }
    fn jacobian(&self, u: &[f64], lambda: &[f64], j: &mut Matrix) {
        let z = u[0] - lambda[0];
        j.resize_zeroed(1, 1);
        j[(0, 0)] = 1.0 / (1.0 + z * z);
    }
}

/// A rank-deficient but **consistent** system: r₀ = u + v − 1, r₁ = 2(u + v − 1).
/// The Jacobian [[1,1],[2,2]] is singular everywhere, yet a whole line of
/// solutions exists. Plain Newton must fail and say why; Levenberg–Marquardt
/// must succeed (the fallback of §6.1).
pub struct SingularConsistentProblem;

impl BifurcationProblem for SingularConsistentProblem {
    fn state_dim(&self) -> usize {
        2
    }
    fn parameter_count(&self) -> usize {
        1
    }
    fn residual(&self, u: &[f64], _lambda: &[f64], r: &mut [f64]) {
        let s = u[0] + u[1] - 1.0;
        r[0] = s;
        r[1] = 2.0 * s;
    }
    fn jacobian(&self, _u: &[f64], _lambda: &[f64], j: &mut Matrix) {
        *j = Matrix::from_row_major(2, 2, &[1.0, 1.0, 2.0, 2.0]);
    }
}

/// A cubic with three equilibria, `du/dt = λ + u − u³`: roots −1, 0, +1 at
/// λ = 0. Newton converges from anywhere, only slowly from far out — a
/// convergence-rate case rather than a globalisation one.
pub struct CubicProblem;

impl BifurcationProblem for CubicProblem {
    fn state_dim(&self) -> usize {
        1
    }
    fn parameter_count(&self) -> usize {
        1
    }
    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
        r[0] = lambda[0] + u[0] - u[0].powi(3);
    }
    fn jacobian(&self, u: &[f64], _lambda: &[f64], j: &mut Matrix) {
        j.resize_zeroed(1, 1);
        j[(0, 0)] = 1.0 - 3.0 * u[0] * u[0];
    }
}
