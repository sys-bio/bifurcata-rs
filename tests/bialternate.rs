//! The bialternate product (specification §6.3, §11.1), ported check by check
//! from the Delphi `Bifurcata.Tests.Bialternate`.
//!
//! §11.1 asks for "a matrix with known μᵢ + μⱼ = 0; normalised determinant
//! vanishes". That is here, but the load-bearing test is stronger: the
//! eigenvalues of 2A ⊙ I must be exactly the pairwise sums μᵢ + μⱼ (i > j) for
//! an arbitrary A, which pins every index of the construction at once.

use bifurcata::bialternate::*;
use bifurcata::complex::Complex;
use bifurcata::linalg::eigenvalues;
use bifurcata::matrix::Matrix;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!((actual - expected).abs() <= tol, "{what}: got {actual}, expected {expected} (tolerance {tol:e})");
}

/// Greedy multiset distance: each expected value consumes its nearest actual.
fn multiset_distance(expected: &[Complex], actual: &[Complex]) -> f64 {
    assert_eq!(expected.len(), actual.len(), "multiset sizes differ");
    let mut used = vec![false; actual.len()];
    let mut worst = 0.0f64;
    for e in expected {
        let (best, d) = actual
            .iter()
            .enumerate()
            .filter(|(j, _)| !used[*j])
            .map(|(j, a)| (j, (*e - *a).modulus()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        used[best] = true;
        worst = worst.max(d);
    }
    worst
}

fn pairwise_sums(values: &[Complex]) -> Vec<Complex> {
    let n = values.len();
    (1..n).flat_map(|p| (0..p).map(move |q| values[p] + values[q])).collect()
}

fn check_spectrum_identity(a: &Matrix, name: &str, tol: f64) {
    let values_a = eigenvalues(a).unwrap();
    let b = bialternate(a);
    assert_eq!(b.rows(), bialternate_dim(a.rows()), "{name}: the bialternate has dimension n(n−1)/2");
    let values_b = eigenvalues(&b).unwrap();
    let d = multiset_distance(&pairwise_sums(&values_a), &values_b);
    close(d, 0.0, tol, &format!("{name}: the spectrum of 2A⊙I is the pairwise sums μᵢ+μⱼ"));
}

#[test]
fn spectrum_is_the_pairwise_sums() {
    // 2×2: the product is 1×1 and equals trace(A).
    let a = Matrix::from_row_major(2, 2, &[1.0, -1.0, 1.0, 1.0]);
    check_spectrum_identity(&a, "2×2", 1e-12);
    close(hopf_test_function(&a), 2.0, 1e-12, "2×2: the test function equals the trace (eigenvalues 1 ± i)");

    #[rustfmt::skip]
    let a = Matrix::from_row_major(3, 3, &[
         0.4, -1.2,  0.7,
         0.9,  0.3, -0.5,
        -0.2,  0.8,  1.1]);
    check_spectrum_identity(&a, "3×3 general", 1e-11);

    // m = 6: every clause of the construction is exercised.
    #[rustfmt::skip]
    let a = Matrix::from_row_major(4, 4, &[
         1.0, -2.0,  0.5,  0.3,
         0.7,  0.2, -1.1,  0.9,
        -0.4,  1.3,  0.6, -0.8,
         0.2, -0.5,  1.7,  0.1]);
    check_spectrum_identity(&a, "4×4 general", 1e-10);

    #[rustfmt::skip]
    let a = Matrix::from_row_major(5, 5, &[
         0.5,  1.0, -0.3,  0.2,  0.9,
        -1.1,  0.4,  0.8, -0.6,  0.1,
         0.3, -0.7,  1.2,  0.5, -0.4,
         0.8,  0.2, -0.9,  0.7,  1.3,
        -0.2,  0.6,  0.4, -1.0,  0.3]);
    check_spectrum_identity(&a, "5×5 general", 1e-9);
}

#[test]
fn zeros_and_their_classification() {
    // A pure Hopf: eigenvalues ±i.
    let a = Matrix::from_row_major(2, 2, &[0.0, -1.0, 1.0, 0.0]);
    let t = hopf_test(&a);
    close(t.value, 0.0, 1e-13, "Hopf 2×2: the test function vanishes");
    assert_eq!(t.sign, 0, "Hopf 2×2: the determinant sign is 0");
    let values = eigenvalues(&a).unwrap();
    let omega = is_hopf_spectrum(&values, 1e-8).expect("Hopf 2×2: the spectrum is classified as Hopf");
    close(omega, 1.0, 1e-12, "Hopf 2×2: ω is 1");
    assert!(!is_neutral_saddle_spectrum(&values, 1e-8), "Hopf 2×2: not mistaken for a neutral saddle");

    // A neutral saddle: +1 and −1 also sum to zero. Recorded, NOT a Hopf (§6.3).
    let a = Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, -1.0]);
    close(hopf_test_function(&a), 0.0, 1e-13, "neutral saddle: the test function vanishes too");
    let values = eigenvalues(&a).unwrap();
    assert!(is_hopf_spectrum(&values, 1e-8).is_none(), "neutral saddle: NOT classified as Hopf");
    assert!(is_neutral_saddle_spectrum(&values, 1e-8), "neutral saddle: classified as a neutral saddle");

    // A Hopf pair beside an unrelated real eigenvalue.
    let a = Matrix::from_row_major(3, 3, &[2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 3.0, 0.0]);
    close(hopf_test_function(&a), 0.0, 1e-11, "3×3 with a Hopf pair: the test function vanishes");
    let omega = is_hopf_spectrum(&eigenvalues(&a).unwrap(), 1e-8).expect("3×3: Hopf pair detected");
    close(omega, 3.0, 1e-11, "3×3: ω is 3");

    // Stable, no pair summing to zero: three negative pairwise sums, sign (−1)³.
    let a = Matrix::from_row_major(3, 3, &[-1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, -3.0]);
    let t = hopf_test(&a);
    assert!(t.value.abs() > 1e-6, "stable matrix: the test function is nonzero");
    assert_eq!(t.sign, -1, "stable 3×3: three negative pairwise sums give sign −1");
    let values = eigenvalues(&a).unwrap();
    assert!(is_hopf_spectrum(&values, 1e-8).is_none(), "stable matrix: no Hopf");
    assert!(!is_neutral_saddle_spectrum(&values, 1e-8), "stable matrix: no neutral saddle");
}

#[test]
fn normalisation_against_overflow() {
    // n = 20, m = 190, pairwise sums 60: |det| = 60^190, far beyond a double.
    let n = 20;
    let mut a = Matrix::zeros(n, n);
    a.add_scaled_identity(30.0);
    let t = hopf_test(&a);
    assert!(t.value.is_finite(), "n = 20: the normalised test function is finite");
    close(t.value, 60.0, 1e-9, "n = 20: the geometric mean of the pivots is exactly 60");
    assert_eq!(t.sign, 1, "n = 20: all pairwise sums positive, sign +1");
    // The raw determinant really is out of range: the normalisation is necessary.
    assert!(t.log_abs_det > 709.0, "n = 20: raw |det| overflows a double (log|det| = {:.1})", t.log_abs_det);
    close(t.log_abs_det, 190.0 * 60f64.ln(), 1e-8, "n = 20: log|det| is m ln 60");

    // And where it would underflow to zero: pairwise sums of 2e-4.
    let mut a = Matrix::zeros(n, n);
    a.add_scaled_identity(1e-4);
    let t = hopf_test(&a);
    assert!(t.value.abs() > 1e-9, "n = 20 small: the normalised value does not underflow to zero");
    close(t.value, 2e-4, 1e-12, "n = 20 small: the geometric mean is exactly 2e-4");
    assert!(t.log_abs_det < -709.0, "n = 20 small: raw |det| would underflow to zero");
}
