//! A dense matrix of `f64`, stored column-major (it wraps `faer::Mat`).
//!
//! Everything in the library that touches faer goes through this module and
//! [`crate::linalg`], so that a change in faer's (pre-1.0) API touches only
//! them. Dense storage throughout is deliberate (§1.3): biochemical models
//! have tens of states, and sparse storage must not leak into interfaces.

use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    m: faer::Mat<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { m: faer::Mat::zeros(rows, cols) }
    }

    pub fn identity(n: usize) -> Self {
        Self { m: faer::Mat::identity(n, n) }
    }

    /// From values in reading order (row by row), the natural way to write a
    /// matrix in source code.
    pub fn from_row_major(rows: usize, cols: usize, values: &[f64]) -> Self {
        assert_eq!(values.len(), rows * cols, "from_row_major: {} values for a {rows}×{cols} matrix", values.len());
        Self { m: faer::Mat::from_fn(rows, cols, |i, j| values[i * cols + j]) }
    }

    pub fn from_fn(rows: usize, cols: usize, f: impl FnMut(usize, usize) -> f64) -> Self {
        Self { m: faer::Mat::from_fn(rows, cols, f) }
    }

    pub fn rows(&self) -> usize {
        self.m.nrows()
    }

    pub fn cols(&self) -> usize {
        self.m.ncols()
    }

    pub fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }

    /// Resize, zeroing every element.
    pub fn resize_zeroed(&mut self, rows: usize, cols: usize) {
        self.m = faer::Mat::zeros(rows, cols);
    }

    pub fn fill(&mut self, value: f64) {
        self.m.fill(value);
    }

    pub fn row(&self, i: usize) -> Vec<f64> {
        (0..self.cols()).map(|j| self[(i, j)]).collect()
    }

    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.rows()).map(|i| self[(i, j)]).collect()
    }

    pub fn transpose(&self) -> Matrix {
        Self { m: self.m.transpose().to_owned() }
    }

    /// `self · other`
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols(), other.rows(), "mul: {}×{} times {}×{}", self.rows(), self.cols(), other.rows(), other.cols());
        Self { m: &self.m * &other.m }
    }

    /// `self · x`
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols(), x.len(), "mul_vec: {} columns, vector of {}", self.cols(), x.len());
        let mut out = vec![0.0; self.rows()];
        for (j, xj) in x.iter().enumerate() {
            if *xj != 0.0 {
                for (i, o) in out.iter_mut().enumerate() {
                    *o += self[(i, j)] * xj;
                }
            }
        }
        out
    }

    /// `selfᵀ · y`
    pub fn mul_transpose_vec(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(self.rows(), y.len(), "mul_transpose_vec: {} rows, vector of {}", self.rows(), y.len());
        (0..self.cols())
            .map(|j| (0..self.rows()).map(|i| self[(i, j)] * y[i]).sum())
            .collect()
    }

    /// `self += alpha · I`
    pub fn add_scaled_identity(&mut self, alpha: f64) {
        for i in 0..self.rows().min(self.cols()) {
            self[(i, i)] += alpha;
        }
    }

    /// The largest absolute column sum.
    pub fn norm_one(&self) -> f64 {
        (0..self.cols())
            .map(|j| (0..self.rows()).map(|i| self[(i, j)].abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    /// The largest absolute row sum.
    pub fn norm_inf(&self) -> f64 {
        (0..self.rows())
            .map(|i| (0..self.cols()).map(|j| self[(i, j)].abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    pub fn norm_frobenius(&self) -> f64 {
        self.m.norm_l2()
    }

    /// The largest absolute element.
    pub fn norm_max(&self) -> f64 {
        self.m.norm_max()
    }

    /// The underlying faer matrix, for [`crate::linalg`].
    pub(crate) fn faer(&self) -> faer::MatRef<'_, f64> {
        self.m.as_ref()
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.m[(i, j)]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.m[(i, j)]
    }
}

// ---- Vector helpers ---------------------------------------------------------------

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot: vectors of different lengths");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}
