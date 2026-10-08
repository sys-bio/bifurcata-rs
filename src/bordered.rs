//! The bordered linear solver (specification §3A.3), the workhorse:
//!
//! ```text
//! [ A   b ] [ x ]   [ f ]
//! [ cᵀ  d ] [ y ] = [ g ]
//! ```
//!
//! where A is n×n and **may be singular** — precisely the case at a fold,
//! where the bordered matrix is still nonsingular.

use crate::linalg::Lu;
use crate::matrix::Matrix;
use crate::types::{SolveCode, SolveStatus};

/// A bordered solve's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct Bordered {
    pub x: Vec<f64>,
    pub y: f64,
}

/// A policy for solving bordered systems (§3A.3 asks for the choice to be a
/// setting, not hard-coded).
pub trait BorderedSolver {
    /// Solve the system. A singular *bordered* matrix is an error; a merely
    /// ill-conditioned one is solved, with a warning left in [`last_status`].
    ///
    /// [`last_status`]: BorderedSolver::last_status
    fn solve(&mut self, a: &Matrix, b: &[f64], c: &[f64], d: f64, f: &[f64], g: f64) -> Result<Bordered, SolveStatus>;

    /// The outcome of the last solve: OK, a failure, or an `IllConditioned`
    /// warning on a solve that nevertheless produced an answer.
    fn last_status(&self) -> &SolveStatus;
}

/// Assembles the full (n+1)×(n+1) matrix and factors it: simple, robust, and
/// correct at folds. The default; at n ≤ 500 the extra cost is irrelevant.
#[derive(Clone, Debug)]
pub struct DirectBordered {
    status: SolveStatus,
    last_rcond: f64,
    /// Below this reciprocal condition number the caller is warned that the
    /// bordering vectors have stopped being a good complement.
    pub rcond_warn_threshold: f64,
}

impl Default for DirectBordered {
    fn default() -> Self {
        Self { status: SolveStatus::ok(), last_rcond: 0.0, rcond_warn_threshold: 1e-12 }
    }
}

impl DirectBordered {
    pub fn new() -> Self {
        Self::default()
    }

    /// The reciprocal condition number of the last bordered matrix (0 if singular).
    pub fn last_rcond(&self) -> f64 {
        self.last_rcond
    }
}

impl BorderedSolver for DirectBordered {
    fn solve(&mut self, a: &Matrix, b: &[f64], c: &[f64], d: f64, f: &[f64], g: f64) -> Result<Bordered, SolveStatus> {
        let fail = |this: &mut Self, status: SolveStatus| {
            this.status = status.clone();
            this.last_rcond = 0.0;
            Err(status)
        };
        if !a.is_square() {
            return fail(self, SolveStatus::fail(SolveCode::InvalidInput, format!("bordered: A is {}×{}, expected square", a.rows(), a.cols())));
        }
        let n = a.rows();
        if b.len() != n || c.len() != n || f.len() != n {
            return fail(
                self,
                SolveStatus::fail(
                    SolveCode::InvalidInput,
                    format!("bordered: expected b, c, f of length {n}; got {}, {}, {}", b.len(), c.len(), f.len()),
                ),
            );
        }

        let mut m = Matrix::zeros(n + 1, n + 1);
        for j in 0..n {
            for i in 0..n {
                m[(i, j)] = a[(i, j)];
            }
        }
        for i in 0..n {
            m[(i, n)] = b[i];
            m[(n, i)] = c[i];
        }
        m[(n, n)] = d;
        let mut rhs = f.to_vec();
        rhs.push(g);

        let lu = match Lu::new(&m) {
            Ok(lu) => lu,
            Err(e) => {
                // A singular A is expected and fine; a singular BORDERED matrix
                // means b and c are no longer a valid complement to A's range.
                return fail(
                    self,
                    SolveStatus::fail(
                        e.code,
                        format!("the bordered matrix is singular (the bordering vectors b, c are no longer a valid complement to the range of A): {}", e.detail),
                    ),
                );
            }
        };
        self.last_rcond = lu.reciprocal_condition();
        let mut solution = lu.solve(&rhs).inspect_err(|e| {
            self.status = e.clone();
        })?;
        let y = solution.pop().unwrap();

        self.status = if self.last_rcond > 0.0 && self.last_rcond < self.rcond_warn_threshold {
            SolveStatus::fail(
                SolveCode::IllConditioned,
                format!(
                    "bordered system solved but rcond = {:.3e} is below the threshold {:.3e}; the result may be meaningless",
                    self.last_rcond, self.rcond_warn_threshold
                ),
            )
        } else {
            SolveStatus::ok()
        };
        Ok(Bordered { x: solution, y })
    }

    fn last_status(&self) -> &SolveStatus {
        &self.status
    }
}
