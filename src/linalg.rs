//! Linear algebra (specification §3A.2): LU with a condition estimate and a
//! signed log-determinant, the eigendecomposition with left and right
//! eigenvectors, null vectors and the SVD.
//!
//! The Delphi version called LAPACK; this one is pure Rust (faer for the
//! eigendecomposition and SVD, our own LU), so that it runs in the browser.

use crate::complex::{self, Complex, ONE, cx};
use crate::matrix::Matrix;
use crate::types::{SolveCode, SolveStatus};

// ---- LU ------------------------------------------------------------------------------

/// An LU factorisation with partial pivoting, `P A = L U`.
///
/// Like LAPACK's `dgetrf`, factoring **completes even when A is singular**:
/// [`Lu::singular`] says so, the solves refuse, and [`Lu::signed_log_det`]
/// reports sign 0 — which matters because a singular matrix is precisely what
/// a test function is looking for. [`Lu::new`] is the convenience that treats
/// singularity as an error.
///
/// The factorisation is a copy; the caller's matrix is untouched and the
/// factors can be reused for several solves, as the PALC step reuses the final
/// Newton factorisation for the new tangent (§5.1).
#[derive(Clone, Debug)]
pub struct Lu {
    lu: Matrix,
    /// `perm[i]` is the row of A that ended up in row i.
    perm: Vec<usize>,
    /// +1 or −1: the parity of the row interchanges.
    parity: i32,
    /// The 1-norm of the original matrix, for the condition estimate.
    norm_one: f64,
    /// The first column with an exactly zero pivot, if any.
    zero_pivot: Option<usize>,
}

impl Lu {
    /// Factor a square matrix; never fails (see [`Lu::singular`]).
    pub fn factor(a: &Matrix) -> Lu {
        assert!(a.is_square(), "LU of a {}×{} matrix", a.rows(), a.cols());
        let n = a.rows();
        let mut lu = a.clone();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut parity = 1;
        let mut zero_pivot = None;
        for k in 0..n {
            let p = (k..n).max_by(|&i, &j| lu[(i, k)].abs().total_cmp(&lu[(j, k)].abs())).unwrap();
            if p != k {
                for j in 0..n {
                    let t = lu[(k, j)];
                    lu[(k, j)] = lu[(p, j)];
                    lu[(p, j)] = t;
                }
                perm.swap(k, p);
                parity = -parity;
            }
            let pivot = lu[(k, k)];
            if pivot == 0.0 {
                // Nothing to eliminate with; carry on, as dgetrf does.
                zero_pivot.get_or_insert(k);
                continue;
            }
            for i in k + 1..n {
                let factor = lu[(i, k)] / pivot;
                lu[(i, k)] = factor;
                if factor != 0.0 {
                    for j in k + 1..n {
                        lu[(i, j)] -= factor * lu[(k, j)];
                    }
                }
            }
        }
        Lu { lu, perm, parity, norm_one: a.norm_one(), zero_pivot }
    }

    /// Factor, treating an exactly singular matrix as an error.
    pub fn new(a: &Matrix) -> Result<Lu, SolveStatus> {
        if !a.is_square() {
            return Err(SolveStatus::fail(SolveCode::InvalidInput, format!("LU of a {}×{} matrix", a.rows(), a.cols())));
        }
        let lu = Lu::factor(a);
        match lu.zero_pivot {
            Some(k) => Err(SolveStatus::fail(SolveCode::SingularJacobian, format!("the matrix is singular (zero pivot in column {k})"))),
            None => Ok(lu),
        }
    }

    pub fn n(&self) -> usize {
        self.perm.len()
    }

    /// True if a pivot was exactly zero.
    pub fn singular(&self) -> bool {
        self.zero_pivot.is_some()
    }

    fn check(&self, b: &[f64]) -> Result<(), SolveStatus> {
        if b.len() != self.n() {
            return Err(SolveStatus::fail(SolveCode::InvalidInput, format!("right-hand side of length {}, matrix of order {}", b.len(), self.n())));
        }
        if let Some(k) = self.zero_pivot {
            return Err(SolveStatus::fail(SolveCode::SingularJacobian, format!("cannot solve: the matrix is singular (zero pivot in column {k})")));
        }
        Ok(())
    }

    /// Solve `A x = b`.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SolveStatus> {
        self.check(b)?;
        let n = self.n();
        let mut x: Vec<f64> = self.perm.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            for j in 0..i {
                x[i] -= self.lu[(i, j)] * x[j];
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= self.lu[(i, j)] * x[j];
            }
            x[i] /= self.lu[(i, i)];
        }
        Ok(x)
    }

    /// Solve `Aᵀ x = b`.
    pub fn solve_transpose(&self, b: &[f64]) -> Result<Vec<f64>, SolveStatus> {
        self.check(b)?;
        let n = self.n();
        // Aᵀ = Uᵀ Lᵀ P: solve Uᵀ z = b, then Lᵀ w = z, then x = Pᵀ w.
        let mut z = b.to_vec();
        for i in 0..n {
            for j in 0..i {
                z[i] -= self.lu[(j, i)] * z[j];
            }
            z[i] /= self.lu[(i, i)];
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                z[i] -= self.lu[(j, i)] * z[j];
            }
        }
        let mut x = vec![0.0; n];
        for (i, &p) in self.perm.iter().enumerate() {
            x[p] = z[i];
        }
        Ok(x)
    }

    /// The determinant as (sign, log|det|), never as one number: the
    /// bialternate product of a 30-state model is 435×435, and the raw product
    /// of its pivots overflows or underflows long before it says anything
    /// useful (§6.3). Sign 0 and log −∞ for a singular matrix.
    pub fn signed_log_det(&self) -> (i32, f64) {
        let mut sign = self.parity;
        let mut log_abs = 0.0;
        for k in 0..self.n() {
            let p = self.lu[(k, k)];
            if p == 0.0 {
                return (0, f64::NEG_INFINITY);
            }
            if p < 0.0 {
                sign = -sign;
            }
            log_abs += p.abs().ln();
        }
        (sign, log_abs)
    }

    /// The determinant itself — for small matrices only.
    pub fn determinant(&self) -> f64 {
        let (sign, log_abs) = self.signed_log_det();
        if sign == 0 { 0.0 } else { sign as f64 * log_abs.exp() }
    }

    /// An estimate of the reciprocal condition number in the 1-norm,
    /// `1 / (‖A‖₁ ‖A⁻¹‖₁)`, as LAPACK's `dgecon` gives; 0 for a singular
    /// matrix. ‖A⁻¹‖₁ is estimated by Hager's method (Higham, *Accuracy and
    /// Stability of Numerical Algorithms*, §15.3), with a few solves.
    pub fn reciprocal_condition(&self) -> f64 {
        let n = self.n();
        if n == 0 || self.singular() || self.norm_one == 0.0 {
            return 0.0;
        }
        let mut x = vec![1.0 / n as f64; n];
        let mut estimate = 0.0;
        for _ in 0..5 {
            let y = self.solve(&x).expect("checked nonsingular");
            let y_norm: f64 = y.iter().map(|v| v.abs()).sum();
            if y_norm <= estimate {
                break;
            }
            estimate = y_norm;
            let sign: Vec<f64> = y.iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect();
            let z = self.solve_transpose(&sign).expect("checked nonsingular");
            let (j, z_max) = z
                .iter()
                .enumerate()
                .map(|(j, v)| (j, v.abs()))
                .fold((0, 0.0), |best, c| if c.1 > best.1 { c } else { best });
            let zx: f64 = z.iter().zip(&x).map(|(a, b)| a * b).sum();
            if z_max <= zx {
                break;
            }
            x = vec![0.0; n];
            x[j] = 1.0;
        }
        if !estimate.is_finite() || estimate == 0.0 {
            return 0.0;
        }
        1.0 / (self.norm_one * estimate)
    }
}

/// The determinant as a signed geometric mean, `sign(det) · |det|^(1/n)`.
///
/// §6.3 forbids a raw determinant as a test function: entries carrying a
/// relative error around 1e-8, accumulated over a few hundred pivots, give a
/// magnitude that can drift or overflow even where the sign is right. This
/// form cannot overflow, is smooth enough for bisection, and crosses zero
/// exactly where the determinant does. Exactly 0 for a singular matrix.
pub fn normalised_determinant(a: &Matrix) -> f64 {
    let n = a.rows();
    if n == 0 {
        return 1.0;
    }
    let (sign, log_abs) = Lu::factor(a).signed_log_det();
    if sign == 0 { 0.0 } else { sign as f64 * (log_abs / n as f64).exp() }
}

// ---- Eigenvalues ------------------------------------------------------------------------

/// The eigenvalues of a real matrix, with right and optionally left eigenvectors.
///
/// Conventions, chosen to match LAPACK's `dgeev` so that nothing downstream
/// depends on which library computed them:
///
/// * a complex conjugate pair is adjacent, **positive imaginary part first**;
/// * the second member of a pair has exactly the conjugate eigenvectors of the
///   first;
/// * each eigenvector has unit 2-norm, and its largest-modulus component is
///   real and positive.
///
/// Right eigenvectors satisfy `A v = μ v`. Left eigenvectors are LAPACK's:
/// `uᴴ A = μ uᴴ`, i.e. `Aᵀ u = conj(μ) u` for real A — the adjoint vector `p`
/// of the normal-form formulas (§6.4). Eigenvector signs (phases) are otherwise
/// arbitrary, in any library: a quantity that depends on one needs a convention.
#[derive(Clone, Debug)]
pub struct Eigen {
    pub values: Vec<Complex>,
    /// `right[j]` belongs to `values[j]`; empty unless requested.
    pub right: Vec<Vec<Complex>>,
    /// `left[j]` belongs to `values[j]`; empty unless requested.
    pub left: Vec<Vec<Complex>>,
}

/// Eigenvalues only, in the order of [`eigen`].
pub fn eigenvalues(a: &Matrix) -> Result<Vec<Complex>, SolveStatus> {
    Ok(eigen(a, false, false)?.values)
}

pub fn eigen(a: &Matrix, want_left: bool, want_right: bool) -> Result<Eigen, SolveStatus> {
    if !a.is_square() {
        return Err(SolveStatus::fail(SolveCode::InvalidInput, format!("eigenvalues of a {}×{} matrix", a.rows(), a.cols())));
    }
    let n = a.rows();
    if n == 0 {
        return Ok(Eigen { values: Vec::new(), right: Vec::new(), left: Vec::new() });
    }
    // No thread pool: these matrices are small (the Delphi version measured
    // OpenBLAS threading costing more than it saved), and the browser has none.
    faer::set_global_parallelism(faer::Par::Seq);

    let decomposition = a
        .faer()
        .eigen()
        .map_err(|e| SolveStatus::fail(SolveCode::LinAlgError, format!("eigendecomposition failed: {e:?}")))?;
    let raw_values: Vec<Complex> = (0..n).map(|j| to_cx(decomposition.S()[j])).collect();
    let raw_right: Vec<Vec<Complex>> = (0..n)
        .map(|j| (0..n).map(|i| to_cx(decomposition.U()[(i, j)])).collect())
        .collect();

    // Order conjugate pairs adjacently, +Im first, as dgeev does.
    let order = pair_order(&raw_values);
    let values: Vec<Complex> = order.iter().map(|&j| raw_values[j]).collect();

    let right = if want_right { conventional_vectors(&values, order.iter().map(|&j| raw_right[j].clone()).collect()) } else { Vec::new() };

    let left = if want_left {
        // Eigenvectors of Aᵀ, matched to conj(μ) for each μ.
        let t = a.transpose().faer().eigen().map_err(|e| SolveStatus::fail(SolveCode::LinAlgError, format!("eigendecomposition of Aᵀ failed: {e:?}")))?;
        let t_values: Vec<Complex> = (0..n).map(|j| to_cx(t.S()[j])).collect();
        let mut used = vec![false; n];
        let mut vectors = Vec::with_capacity(n);
        for mu in &values {
            let target = mu.conjugate();
            let k = (0..n)
                .filter(|&k| !used[k])
                .min_by(|&i, &j| (t_values[i] - target).modulus().total_cmp(&(t_values[j] - target).modulus()))
                .unwrap();
            used[k] = true;
            vectors.push((0..n).map(|i| to_cx(t.U()[(i, k)])).collect());
        }
        conventional_vectors(&values, vectors)
    } else {
        Vec::new()
    };

    Ok(Eigen { values, right, left })
}

fn to_cx(z: faer::c64) -> Complex {
    cx(z.re, z.im)
}

/// An order putting each conjugate pair together, positive imaginary part first.
fn pair_order(values: &[Complex]) -> Vec<usize> {
    let n = values.len();
    let mut used = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for j in 0..n {
        if used[j] {
            continue;
        }
        let v = values[j];
        if v.im == 0.0 {
            used[j] = true;
            order.push(j);
            continue;
        }
        // The partner: the unused value closest to the conjugate.
        let partner = (0..n)
            .filter(|&k| k != j && !used[k] && values[k].im != 0.0 && values[k].im.signum() != v.im.signum())
            .min_by(|&a, &b| (values[a] - v.conjugate()).modulus().total_cmp(&(values[b] - v.conjugate()).modulus()));
        used[j] = true;
        match partner {
            Some(k) => {
                used[k] = true;
                let (first, second) = if v.im > 0.0 { (j, k) } else { (k, j) };
                order.push(first);
                order.push(second);
            }
            None => order.push(j),
        }
    }
    order
}

/// Apply the conventions: unit norm, largest component real and positive, and
/// the second member of each conjugate pair exactly conjugate to the first.
fn conventional_vectors(values: &[Complex], mut vectors: Vec<Vec<Complex>>) -> Vec<Vec<Complex>> {
    for v in vectors.iter_mut() {
        normalise_phase(v);
    }
    for j in 1..values.len() {
        if values[j].im < 0.0 && values[j - 1].im > 0.0 && (values[j] - values[j - 1].conjugate()).modulus() <= 1e-12 * values[j].modulus().max(1.0) {
            vectors[j] = vectors[j - 1].iter().map(|z| z.conjugate()).collect();
        }
    }
    vectors
}

/// Unit 2-norm, with the largest-modulus component real and positive (the
/// lowest index wins a tie).
fn normalise_phase(v: &mut [Complex]) {
    let norm = complex::norm2(v);
    if norm == 0.0 {
        return;
    }
    let (k, _) = v.iter().enumerate().fold((0, -1.0f64), |best, (i, z)| if z.modulus() > best.1 + 1e-14 * best.1.abs() { (i, z.modulus()) } else { best });
    let phase = v[k] / cx(v[k].modulus(), 0.0);
    let factor = ONE / (phase * cx(norm, 0.0));
    complex::scale(v, factor);
    v[k].im = 0.0;
}

/// The number of eigenvalues with Re μ > `tol` (§6.2).
pub fn unstable_dimension(values: &[Complex], tol: f64) -> usize {
    values.iter().filter(|z| z.re > tol).count()
}

/// Scale `p` so that `⟨p, q⟩ = 1` (Hermitian, conjugate-linear in `p`).
/// Returns false if `⟨p, q⟩` is numerically zero.
pub fn normalise_adjoint(p: &mut [Complex], q: &[Complex]) -> bool {
    let ip = complex::dot(p, q);
    if ip.modulus() < 1e-300 {
        return false;
    }
    // ⟨p, q⟩ is conjugate-linear in p, so dividing p by conj(⟨p, q⟩) makes it exactly 1.
    complex::scale(p, ONE / ip.conjugate());
    true
}

// ---- SVD and null vectors ---------------------------------------------------------------

/// The full SVD `A = U · diag(s) · Vᵀ`, with U (m×m) and Vᵀ (n×n), singular
/// values in decreasing order. Needed where both null spaces matter: the
/// algebraic branching equation (§6.4) spans tangents with the right null
/// space and projects with the left one.
pub fn svd_full(a: &Matrix) -> Result<(Matrix, Vec<f64>, Matrix), SolveStatus> {
    faer::set_global_parallelism(faer::Par::Seq);
    let svd = a.faer().svd().map_err(|e| SolveStatus::fail(SolveCode::LinAlgError, format!("SVD failed: {e:?}")))?;
    let (m, n) = (a.rows(), a.cols());
    let k = m.min(n);
    let mut s: Vec<(usize, f64)> = (0..k).map(|i| (i, svd.S()[i])).collect();
    // Decreasing order, whatever order faer returns them in.
    s.sort_by(|a, b| b.1.total_cmp(&a.1));
    let order: Vec<usize> = s.iter().map(|(i, _)| *i).chain(k..n.max(m)).collect();
    let u = Matrix::from_fn(m, m, |i, j| svd.U()[(i, if j < k { order[j] } else { j })]);
    let vt = Matrix::from_fn(n, n, |i, j| svd.V()[(j, if i < k { order[i] } else { i })]);
    Ok((u, s.iter().map(|(_, v)| *v).collect(), vt))
}

/// The right null vector of an m×n matrix (m < n expected): the right singular
/// vector of the smallest singular value, normalised. Used for the initial
/// tangent (§6.1). Returns the vector and the smallest singular value (0 when
/// m < n, where the null space exists by construction).
pub fn null_vector(a: &Matrix) -> Result<(Vec<f64>, f64), SolveStatus> {
    let (m, n) = (a.rows(), a.cols());
    if m == 0 || n == 0 {
        return Err(SolveStatus::fail(SolveCode::InvalidInput, "null vector of an empty matrix"));
    }
    let (_, s, vt) = svd_full(a)?;
    let v = vt.row(n - 1);
    let smallest = if m >= n { s[n - 1] } else { 0.0 };
    Ok((v, smallest))
}
