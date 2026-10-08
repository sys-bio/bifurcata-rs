//! The bialternate product `2A ⊙ I` and the Hopf test function built on it
//! (specification §6.3).
//!
//! For an n×n matrix A the product has dimension m = n(n−1)/2, indexed by the
//! pairs (p, q) with p > q, and is singular exactly when A has two eigenvalues
//! summing to zero, `μᵢ + μⱼ = 0`. That includes a Hopf pair ±iω, but **also**
//! a real pair ±σ — a neutral saddle, which is not a bifurcation. The
//! determinant cannot tell them apart; the spectrum can
//! ([`is_hopf_spectrum`], [`is_neutral_saddle_spectrum`]).
//!
//! Kuznetsov §10.2.2 gives the construction. With pairs (p, q), p > q and
//! (r, s), r > s:
//!
//! ```text
//! (2A ⊙ I)[(p,q),(r,s)] = −A[p,s]          if r = q
//!                          A[p,r]          if r ≠ p and s = q
//!                          A[p,p] + A[q,q] if r = p and s = q
//!                          A[q,s]          if r = p and s ≠ q
//!                         −A[q,r]          if s = p
//!                          0               otherwise
//! ```
//!
//! It is index-fiddly, so the tests check the strongest property available:
//! the spectrum of the product is exactly the pairwise sums `μ_p + μ_q`.

use crate::complex::Complex;
use crate::linalg::Lu;
use crate::matrix::Matrix;

/// n(n−1)/2, the dimension of the bialternate product of an n×n matrix.
pub fn bialternate_dim(n: usize) -> usize {
    n * n.saturating_sub(1) / 2
}

/// The pairs (p, q), p > q, in the order the rows and columns use.
fn pairs(n: usize) -> Vec<(usize, usize)> {
    (1..n).flat_map(|p| (0..p).map(move |q| (p, q))).collect()
}

/// Build `2A ⊙ I`.
pub fn bialternate(a: &Matrix) -> Matrix {
    assert!(a.is_square(), "bialternate: matrix is {}×{}, expected square", a.rows(), a.cols());
    let pairs = pairs(a.rows());
    let m = pairs.len();
    let mut b = Matrix::zeros(m, m);
    for (i, &(p, q)) in pairs.iter().enumerate() {
        for (j, &(r, s)) in pairs.iter().enumerate() {
            // Ordered so the r = p, s = q case is reached before the more
            // general r = p or s = q ones.
            b[(i, j)] = if r == q {
                -a[(p, s)]
            } else if r == p && s == q {
                a[(p, p)] + a[(q, q)]
            } else if s == p {
                -a[(q, r)]
            } else if s == q {
                a[(p, r)]
            } else if r == p {
                a[(q, s)]
            } else {
                0.0
            };
        }
    }
    b
}

/// The Hopf test function and the raw pair it is made from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HopfTest {
    /// `sign(det) · |det|^(1/m)`: the signed geometric mean of the pivots.
    /// Smooth, cannot overflow, and crosses zero exactly where the determinant
    /// does — all bisection needs. 0 when the product is exactly singular.
    pub value: f64,
    /// The sign of the determinant: the part that carries the bifurcation.
    pub sign: i32,
    /// log |det|; −∞ when singular.
    pub log_abs_det: f64,
}

/// The Hopf test function of §6.3, normalised as it must be: with A built from
/// numerically differentiated entries, a raw 435×435 determinant (n = 30) can
/// drift in magnitude, overflow or underflow even where its sign is right.
pub fn hopf_test(a: &Matrix) -> HopfTest {
    let m = bialternate_dim(a.rows());
    if m == 0 {
        // n < 2: no pair of eigenvalues can sum to zero.
        return HopfTest { value: 1.0, sign: 1, log_abs_det: 0.0 };
    }
    let (sign, log_abs_det) = Lu::factor(&bialternate(a)).signed_log_det();
    let value = if sign == 0 { 0.0 } else { sign as f64 * (log_abs_det / m as f64).exp() };
    HopfTest { value, sign, log_abs_det }
}

/// [`hopf_test`]'s value alone.
pub fn hopf_test_function(a: &Matrix) -> f64 {
    hopf_test(a).value
}

/// A conjugate pair on the imaginary axis to within `tol` — a genuine Hopf
/// point rather than a neutral saddle. Returns ω, the positive imaginary part;
/// where several pairs qualify, the one closest to the axis.
pub fn is_hopf_spectrum(values: &[Complex], tol: f64) -> Option<f64> {
    values
        .iter()
        .filter(|z| z.im > tol && z.re.abs() <= tol)
        .min_by(|a, b| a.re.abs().total_cmp(&b.re.abs()))
        .map(|z| z.im)
}

/// A real pair μ, −μ with μ bounded away from zero: a neutral saddle. Recorded,
/// but not a bifurcation.
pub fn is_neutral_saddle_spectrum(values: &[Complex], tol: f64) -> bool {
    for (i, a) in values.iter().enumerate() {
        if a.im.abs() > tol || a.re.abs() <= tol {
            // Complex, or a zero rather than half of a pair.
            continue;
        }
        for b in &values[i + 1..] {
            if b.im.abs() <= tol && (a.re + b.re).abs() <= tol * a.re.abs().max(1.0) {
                return true;
            }
        }
    }
    false
}
