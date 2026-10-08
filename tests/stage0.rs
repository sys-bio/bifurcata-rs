//! Stage 0 (specification §11.1), ported check by check from the Delphi
//! `Bifurcata.Tests.Stage0`: complex arithmetic, the matrix, LU, eigenvectors
//! and their unpacking, the bordered solver with singular A, null vectors, and
//! the types whose names are the wire format.
//!
//! Dropped from the Delphi suite: the checks that the raw buffer handed to
//! LAPACK is column-major and on its leading dimension. There is no LAPACK
//! here; the storage is faer's and not exposed.

use bifurcata::bordered::{BorderedSolver, DirectBordered};
use bifurcata::complex::{self, Complex, I, ONE, ZERO, cx};
use bifurcata::linalg::{Lu, eigen, normalise_adjoint, null_vector, unstable_dimension};
use bifurcata::matrix::{self, Matrix};
use bifurcata::types::*;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!((actual - expected).abs() <= tol, "{what}: got {actual}, expected {expected} (tolerance {tol:e})");
}

// ---- Complex arithmetic -------------------------------------------------------

#[test]
fn complex_arithmetic() {
    let a = cx(1.0, 2.0);
    let b = cx(3.0, -1.0);

    // (1+2i)(3−i) = 3 − i + 6i − 2i² = 5 + 5i
    let c = a * b;
    close(c.re, 5.0, 1e-15, "multiply: real part");
    close(c.im, 5.0, 1e-15, "multiply: imaginary part");

    let r = c / b;
    close(r.re, a.re, 1e-14, "divide inverts multiply: real part");
    close(r.im, a.im, 1e-14, "divide inverts multiply: imaginary part");

    close(cx(3.0, 4.0).modulus(), 5.0, 1e-15, "modulus of 3+4i is 5");
    close(cx(3.0, 4.0).modulus_squared(), 25.0, 1e-15, "modulus squared");

    let c = a * a.conjugate();
    close(c.re, a.modulus_squared(), 1e-15, "z conj(z) is real");
    close(c.im, 0.0, 1e-15, "z conj(z) has zero imaginary part");

    let c = a * a.reciprocal();
    close(c.re, 1.0, 1e-14, "z (1/z) real part is 1");
    close(c.im, 0.0, 1e-14, "z (1/z) imaginary part is 0");

    let c = I * I;
    close(c.re, -1.0, 1e-15, "i² is −1");
    close(c.im, 0.0, 1e-15, "i² has zero imaginary part");

    let c = a * (b + ONE);
    let r = a * b + a;
    close(c.re, r.re, 1e-14, "distributive law: real part");
    close(c.im, r.im, 1e-14, "distributive law: imaginary part");

    let c: Complex = 2.0.into();
    close(c.re, 2.0, 1e-15, "from f64 sets re");
    close(c.im, 0.0, 1e-15, "from f64 zeroes im");

    // Smith's algorithm: a naive |b|² denominator would overflow here.
    let c = cx(1e200, 0.0) / cx(1e200, 1e200);
    assert!(c.is_finite(), "divide by a huge complex does not overflow");
    close(c.re, 0.5, 1e-12, "huge divide: real part");
    close(c.im, -0.5, 1e-12, "huge divide: imaginary part");

    close(cx(0.0, 1.0).argument(), std::f64::consts::FRAC_PI_2, 1e-15, "arg(i) is π/2");

    // The Hermitian product is conjugate-linear in its FIRST argument.
    let p = [cx(1.0, 1.0), cx(2.0, -1.0)];
    let q = [cx(0.0, 1.0), cx(1.0, 0.0)];
    let ip = complex::dot(&p, &q);
    // conj(1+i)·i + conj(2−i)·1 = (1−i)i + (2+i) = 3 + 2i
    close(ip.re, 3.0, 1e-14, "dot: real part");
    close(ip.im, 2.0, 1e-14, "dot: imaginary part");
    let ip = complex::dot(&p, &p);
    close(ip.im, 0.0, 1e-14, "dot(p, p) is real");
    close(ip.re, complex::norm2(&p).powi(2), 1e-13, "dot(p, p) is the squared 2-norm");
}

// ---- The matrix -------------------------------------------------------------------

#[test]
fn matrix_basics() {
    // from_row_major takes reading order.
    let a = Matrix::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(a[(0, 0)], 1.0, "element (0,0)");
    assert_eq!(a[(0, 2)], 3.0, "element (0,2)");
    assert_eq!(a[(1, 0)], 4.0, "element (1,0)");
    assert_eq!(a[(1, 2)], 6.0, "element (1,2)");

    let w = a.mul_vec(&[1.0, 1.0, 1.0]);
    close(w[0], 6.0, 1e-15, "A·ones row 0");
    close(w[1], 15.0, 1e-15, "A·ones row 1");

    let w = a.mul_transpose_vec(&[1.0, 0.0]);
    close(w[0], 1.0, 1e-15, "Aᵀ·e0 element 0");
    close(w[2], 3.0, 1e-15, "Aᵀ·e0 element 2");

    let b = a.transpose();
    assert_eq!(b.rows(), 3, "the transpose has 3 rows");
    assert_eq!(b[(2, 0)], a[(0, 2)], "transpose swaps indices");

    let c = a.mul(&b); // 2×3 · 3×2 = 2×2
    assert_eq!(c.rows(), 2, "the product has 2 rows");
    close(c[(0, 0)], 14.0, 1e-14, "A·Aᵀ element (0,0)"); // 1+4+9
    close(c[(0, 1)], 32.0, 1e-14, "A·Aᵀ element (0,1)"); // 4+10+18

    close(a.norm_one(), 9.0, 1e-15, "1-norm is the largest column sum (3+6)");
    close(a.norm_inf(), 15.0, 1e-15, "∞-norm is the largest row sum (4+5+6)");
    close(a.norm_frobenius(), 91f64.sqrt(), 1e-14, "Frobenius norm");

    let mut id = Matrix::identity(3);
    assert_eq!(id[(1, 1)], 1.0, "identity diagonal");
    assert_eq!(id[(1, 2)], 0.0, "identity off-diagonal");
    id.add_scaled_identity(-1.0);
    assert_eq!(id.norm_max(), 0.0, "I − I is zero");

    close(matrix::norm2(&[3.0, 4.0]), 5.0, 1e-15, "vector 2-norm");
    close(matrix::norm_inf(&[3.0, -7.0]), 7.0, 0.0, "vector ∞-norm");
}

// ---- LU -------------------------------------------------------------------------------

#[test]
fn lu_factorisation() {
    // det [[1,2],[3,4]] = −2
    let a = Matrix::from_row_major(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let lu = Lu::new(&a).expect("factor a nonsingular 2×2");
    close(lu.determinant(), -2.0, 1e-13, "the determinant is −2");
    let (sign, log_abs) = lu.signed_log_det();
    assert_eq!(sign, -1, "signed log det: the sign is negative");
    close(log_abs, 2f64.ln(), 1e-13, "signed log det: log|det|");

    // A x = b with x = (1,1) gives b = (3,7).
    let x = lu.solve(&[3.0, 7.0]).expect("solve succeeds");
    close(x[0], 1.0, 1e-13, "solution component 0");
    close(x[1], 1.0, 1e-13, "solution component 1");

    // Aᵀ x = b with x = (1,1) gives b = (4,6).
    let x = lu.solve_transpose(&[4.0, 6.0]).expect("transpose solve succeeds");
    close(x[0], 1.0, 1e-13, "transpose solution component 0");
    close(x[1], 1.0, 1e-13, "transpose solution component 1");
    assert!(lu.reciprocal_condition() > 0.0, "rcond is positive for a well-conditioned matrix");

    // det [[0,1],[1,0]] = −1, reached only through a row swap: the parity carries the sign.
    let p = Lu::new(&Matrix::from_row_major(2, 2, &[0.0, 1.0, 1.0, 0.0])).expect("factor a permutation matrix");
    close(p.determinant(), -1.0, 1e-14, "the permutation determinant is −1 (parity tracked)");

    let d = Lu::new(&Matrix::from_row_major(3, 3, &[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0])).unwrap();
    close(d.determinant(), 24.0, 1e-12, "the diagonal determinant is 24");

    // Exactly singular: reported, not garbage, and the determinant's sign is 0.
    let s = Matrix::from_row_major(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    let err = Lu::new(&s).expect_err("a singular matrix is reported as a failure");
    assert_eq!(err.code, SolveCode::SingularJacobian, "a singular matrix gives SingularJacobian");
    let factored = Lu::factor(&s);
    assert!(factored.singular());
    assert_eq!(factored.signed_log_det().0, 0, "a singular matrix has determinant sign 0");
    assert!(factored.solve(&[1.0, 1.0]).is_err(), "solving with a singular factorisation is refused");
}

// ---- Eigenvalues and eigenvectors -------------------------------------------------------

/// max_i |(A v − μ v)_i|: validates a right eigenvector independently of its scaling.
fn eigen_residual(a: &Matrix, v: &[Complex], mu: Complex) -> f64 {
    let n = a.rows();
    (0..n)
        .map(|i| {
            let s = (0..n).fold(ZERO, |s, j| s + Complex::from(a[(i, j)]) * v[j]);
            (s - mu * v[i]).modulus()
        })
        .fold(0.0, f64::max)
}

/// LAPACK's left eigenvectors satisfy uᴴ A = μ uᴴ, i.e. Aᵀ p = conj(μ) p for
/// real A — the adjoint relation of §6.4.
fn adjoint_residual(a: &Matrix, p: &[Complex], mu: Complex) -> f64 {
    let n = a.rows();
    (0..n)
        .map(|i| {
            let s = (0..n).fold(ZERO, |s, j| s + Complex::from(a[(j, i)]) * p[j]);
            (s - mu.conjugate() * p[i]).modulus()
        })
        .fold(0.0, f64::max)
}

#[test]
fn eigenvectors_and_their_conventions() {
    // [[0,−1],[1,0]] has eigenvalues exactly +i and −i.
    let a = Matrix::from_row_major(2, 2, &[0.0, -1.0, 1.0, 0.0]);
    let e = eigen(&a, true, true).expect("eigen succeeds on the rotation matrix");
    assert_eq!(e.values.len(), 2, "two eigenvalues");
    for (j, v) in e.values.iter().enumerate() {
        close(v.re, 0.0, 1e-14, &format!("eigenvalue {j} has zero real part"));
    }
    assert!(e.values.iter().any(|v| (v.im - 1.0).abs() < 1e-14) && e.values.iter().any(|v| (v.im + 1.0).abs() < 1e-14), "the spectrum is exactly ±i");
    // dgeev's convention, kept here: a conjugate pair comes positive-imaginary first.
    assert!(e.values[0].im > 0.0, "the pair is ordered positive-imaginary first");
    assert!(e.values[1].im < 0.0, "the second member has negative imaginary part");

    for j in 0..2 {
        close(eigen_residual(&a, &e.right[j], e.values[j]), 0.0, 1e-13, &format!("right eigenvector {j} satisfies A v = μ v"));
        close(adjoint_residual(&a, &e.left[j], e.values[j]), 0.0, 1e-13, &format!("left eigenvector {j} satisfies Aᵀ p = conj(μ) p"));
    }

    // The two members of the pair have conjugate eigenvectors.
    close((e.right[0][0].re - e.right[1][0].re).abs(), 0.0, 1e-14, "paired eigenvectors share a real part");
    close((e.right[0][0].im + e.right[1][0].im).abs(), 0.0, 1e-14, "paired eigenvectors have opposite imaginary parts");

    // ⟨p, q⟩ = 1, as every normal-form formula uses.
    let q = e.right[0].clone();
    let mut p = e.left[0].clone();
    assert!(normalise_adjoint(&mut p, &q), "adjoint normalisation succeeds");
    let ip = complex::dot(&p, &q);
    close(ip.re, 1.0, 1e-12, "normalised ⟨p,q⟩ real part is 1");
    close(ip.im, 0.0, 1e-12, "normalised ⟨p,q⟩ imaginary part is 0");

    // A real eigenvalue right next to a conjugate pair: block diagonal [2] and
    // [[1,−1],[1,1]] gives 2, 1+i, 1−i.
    let a = Matrix::from_row_major(3, 3, &[2.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0]);
    let e = eigen(&a, true, true).expect("eigen succeeds on the mixed 3×3");
    assert_eq!(e.values.iter().filter(|v| v.im.abs() > 1e-12).count(), 2, "exactly two complex eigenvalues");
    for j in 0..3 {
        close(eigen_residual(&a, &e.right[j], e.values[j]), 0.0, 1e-13, &format!("mixed 3×3: right eigenvector {j}"));
        close(adjoint_residual(&a, &e.left[j], e.values[j]), 0.0, 1e-13, &format!("mixed 3×3: left eigenvector {j}"));
    }
    assert_eq!(unstable_dimension(&e.values, 0.0), 3, "all three eigenvalues have positive real part");

    let stable = eigen(&Matrix::from_row_major(2, 2, &[-1.0, 0.0, 0.0, -2.0]), false, false).unwrap();
    assert_eq!(unstable_dimension(&stable.values, 0.0), 0, "a stable matrix has unstable dimension 0");
    assert!(stable.right.is_empty() && stable.left.is_empty(), "eigenvectors only when asked for");
}

/// The normalisation convention: unit norm, largest component real and
/// positive — so eigenvector signs are reproducible (CLAUDE.md: "eigenvector
/// signs are arbitrary … any quantity carrying one needs a convention").
#[test]
fn eigenvector_normalisation_convention() {
    let a = Matrix::from_row_major(3, 3, &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
    let e = eigen(&a, true, true).unwrap();
    for v in e.right.iter().chain(&e.left) {
        close(complex::norm2(v), 1.0, 1e-13, "unit norm");
        // The first of the largest components (the lowest index wins a tie).
        let biggest = v.iter().map(|z| z.modulus()).fold(0.0, f64::max);
        let largest = *v.iter().find(|z| z.modulus() >= biggest * (1.0 - 1e-12)).unwrap();
        assert!(largest.re > 0.0 && largest.im == 0.0, "largest component real and positive: {largest:?}");
    }
}

// ---- The bordered solver -----------------------------------------------------------------

#[test]
fn bordered_solver_with_singular_a() {
    let mut solver = DirectBordered::new();

    // A is exactly singular (rank 1, null vector (1,−1)): the fold case of
    // §3A.3. A alone cannot be factored, but the bordered matrix can.
    let a = Matrix::from_row_major(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(a[(0, 0)] * a[(1, 1)] - a[(0, 1)] * a[(1, 0)], 0.0, "the test matrix A is genuinely singular");
    // Built so the answer is x = (1,2), y = 3:
    //   f = A x + b y = (3,3) + (3,−3) = (6,0)
    //   g = c·x + d y = (1 − 2) + 0    = −1
    let s = solver
        .solve(&a, &[1.0, -1.0], &[1.0, -1.0], 0.0, &[6.0, 0.0], -1.0)
        .expect("the bordered solve succeeds even though A is singular");
    close(s.x[0], 1.0, 1e-12, "bordered solution x[0]");
    close(s.x[1], 2.0, 1e-12, "bordered solution x[1]");
    close(s.y, 3.0, 1e-12, "bordered solution y");
    assert!(solver.last_rcond() > 1e-14, "the bordered system is well conditioned despite A being singular");
    assert!(solver.last_status().is_ok());

    // A degenerate bordering: b in A's range and c orthogonal to its null
    // vector, so the bordered matrix is itself singular. Reported, not silent.
    let err = solver
        .solve(&a, &[1.0, 1.0], &[1.0, 1.0], 0.0, &[1.0, 1.0], 1.0)
        .expect_err("a degenerate bordering is reported as a failure");
    assert!(!err.detail.is_empty(), "the failure carries a diagnostic");
    assert!(!solver.last_status().detail.is_empty(), "and so does last_status");

    // A dimension mismatch is caught rather than reading past the end.
    let err = solver
        .solve(&Matrix::identity(2), &[0.0; 3], &[0.0; 2], 0.0, &[0.0; 2], 0.0)
        .expect_err("a mismatched border length is rejected");
    assert_eq!(err.code, SolveCode::InvalidInput, "a mismatched border gives InvalidInput");
    assert_eq!(solver.last_status().code, SolveCode::InvalidInput);
}

// ---- The null vector -------------------------------------------------------------------------

#[test]
fn null_vector_for_the_initial_tangent() {
    // 2×3, the shape of [F_u F_λ] for a 2-state model; its null space is (1,−2,1).
    let a = Matrix::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let (v, smallest) = null_vector(&a).expect("null_vector succeeds");
    assert!(smallest >= 0.0);
    assert_eq!(v.len(), 3, "the null vector has the column count");
    close(matrix::norm2(&v), 1.0, 1e-13, "the null vector is normalised");
    close(matrix::norm_inf(&a.mul_vec(&v)), 0.0, 1e-13, "A v is zero");
    let scale = v[0];
    assert!(scale.abs() > 1e-13, "the null vector unexpectedly has a zero first component");
    close(v[1] / scale, -2.0, 1e-11, "null direction component 1");
    close(v[2] / scale, 1.0, 1e-11, "null direction component 2");
}

// ---- Types ------------------------------------------------------------------------------------

#[test]
fn types_and_wire_names() {
    let mut nf = NormalForm::new();
    nf.set("l1", -0.0625);
    nf.set("omega", 1.0);
    assert_eq!(nf.len(), 2, "two coefficients stored");
    assert_eq!(nf.get("l1"), Some(-0.0625), "l1 round-trips");
    nf.set("l1", -0.1);
    assert_eq!(nf.len(), 2, "overwriting does not add a duplicate");
    assert_eq!(nf.get_or("l1", 0.0), -0.1, "l1 was overwritten");
    assert!(!nf.contains("a"), "an absent coefficient reports absent");

    // Every kind must survive the string round trip, or stored baselines break.
    for k in BifurcationKind::ALL {
        assert_eq!(BifurcationKind::parse(k.as_str()), k, "kind round-trips: {}", k.as_str());
    }
    assert_eq!(BifurcationKind::Hopf.abbreviation(), "H", "Hopf abbreviates to H");
    assert_eq!(TerminationReason::ParameterBound.as_str(), "parameterBound", "termination wire name");
    for t in TerminationReason::ALL {
        assert_eq!(TerminationReason::parse(t.as_str()), Some(t), "termination round-trips: {}", t.as_str());
    }

    // §10.1: a bound is exit code 1, giving up at a fold is 2.
    assert_eq!(TerminationReason::ParameterBound.exit_code(), 1, "a parameter bound exits 1");
    assert_eq!(TerminationReason::CorrectorFailed.exit_code(), 2, "a corrector failure exits 2");
    assert_eq!(TerminationReason::MaxPoints.exit_code(), 0, "reaching the point budget exits 0");

    // A branch owns its points: changing the caller's copy afterwards does not
    // reach the stored one. (In Delphi this needed care — the engine reuses one
    // working point every step; in Rust ownership guarantees it.)
    let mut p = CurvePoint::new(2, 1);
    p.u = vec![1.0, 2.0];
    p.lambda[0] = 0.5;
    let mut branch = Branch::new(0);
    branch.add_point(p.clone());
    p.u[0] = 99.0;
    branch.add_point(p.clone());
    assert_eq!(branch.points.len(), 2, "two points stored");
    assert_eq!(branch.points[0].u[0], 1.0, "the first stored point did not alias the mutated array");
    assert_eq!(branch.points[1].u[0], 99.0, "the second stored point took the new value");
    assert_eq!(branch.parameter_range(0), Some((0.5, 0.5)));
}
