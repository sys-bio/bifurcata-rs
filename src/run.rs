//! An equilibrium run as the console harness's `run` command does it (§10.1):
//! continue from a start point in one or both directions, each into its own
//! branch, and summarise the outcome as an exit code and a message.

use crate::continuation::{ContinuationEngine, ContinuationOptions, Declined};
use crate::equilibrium::EquilibriumCurve;
use crate::problem::BifurcationProblem;
use crate::serialise::RunMetadata;
use crate::types::*;

/// Exit codes of the console harness (§10.1).
pub const EXIT_OK: i32 = 0;
pub const EXIT_BOUND: i32 = 1;
pub const EXIT_CONTINUATION: i32 = 2;
pub const EXIT_NO_START: i32 = 3;
pub const EXIT_MODEL_LOAD: i32 = 4;
pub const EXIT_COMPARE_MISMATCH: i32 = 5;
pub const EXIT_USAGE: i32 = 64;

pub struct EquilibriumRun<'a> {
    pub problem: &'a dyn BifurcationProblem,
    pub model_name: String,
    pub model_source: String,
    /// Every parameter's value; the active one's is the starting value.
    pub lambda0: Vec<f64>,
    pub active: usize,
    /// The starting equilibrium (independent states).
    pub u0: Vec<f64>,
    /// +1, −1, or 0 for both.
    pub direction: i32,
    pub options: ContinuationOptions,
}

/// The branches and the metadata that describes them, or a failure to start.
pub struct RunResult {
    pub branches: Vec<Branch>,
    pub meta: RunMetadata,
    /// Located zeros the defining system declined, with its reasons.
    pub declined: Vec<Declined>,
}

fn direction_name(d: i32) -> &'static str {
    if d > 0 { "+" } else { "-" }
}

impl EquilibriumRun<'_> {
    pub fn run(&self) -> Result<RunResult, SolveStatus> {
        let p = self.problem;
        let curve = EquilibriumCurve::new(p, &self.lambda0, self.active, &self.options)?;
        let start = self.lambda0[self.active];
        let x0 = curve.pack(&self.u0, start);

        let mut meta = RunMetadata {
            model_name: self.model_name.clone(),
            model_source: self.model_source.clone(),
            state_names: (0..p.state_dim()).map(|i| p.state_name(i)).collect(),
            independent_count: p.state_dim(),
            active_parameter_names: vec![p.parameter_name(self.active)],
            active_parameter_indices: vec![self.active],
            range_lo: self.options.parameter_min,
            range_hi: self.options.parameter_max,
            direction: self.direction,
            exit_code: EXIT_OK,
            message: String::new(),
        };

        // Both directions by default: a branch with folds has features on both
        // sides of wherever the run starts, and one direction shows only one
        // side. Reporting half is not a partial answer, it is a wrong one.
        let directions: Vec<i32> = if self.direction == 0 { vec![1, -1] } else { vec![self.direction.signum()] };
        let mut branches = Vec::new();
        let declined = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        for (d, &dir) in directions.iter().enumerate() {
            let mut engine = ContinuationEngine::with_branch(&curve, self.options.clone(), Branch::new(d));
            let sink = std::rc::Rc::clone(&declined);
            engine.on_declined(move |x| sink.borrow_mut().push(x.clone()));
            engine.initialise(&x0, dir).map_err(|e| SolveStatus {
                code: e.code,
                detail: format!("the starting point could not be corrected onto the curve (direction {}): {}", direction_name(dir), e.detail),
                ..e
            })?;
            engine.run();
            let mut branch = engine.into_branch();
            branch.state_names = meta.state_names.clone();
            branch.parameter_names = (0..p.parameter_count()).map(|k| p.parameter_name(k)).collect();

            // The run as a whole is only as good as its worst direction.
            match branch.termination {
                TerminationReason::ParameterBound | TerminationReason::StateBound => {
                    if meta.exit_code == EXIT_OK {
                        meta.exit_code = EXIT_BOUND;
                        meta.message = format!("completed at {}", branch.termination.as_str());
                    }
                }
                TerminationReason::StepTooSmall | TerminationReason::CorrectorFailed => {
                    meta.exit_code = EXIT_CONTINUATION;
                    meta.message = format!("continuation failed (direction {}): {}", direction_name(dir), branch.termination_detail);
                }
                TerminationReason::MaxPoints => {
                    // Not a failure, but truncated rather than finished.
                    meta.message = format!(
                        "stopped at the {}-point limit (direction {}); the curve is truncated, not finished - raise maxPoints or --max-points",
                        self.options.max_points,
                        direction_name(dir)
                    );
                }
                _ => {
                    if meta.message.is_empty() {
                        meta.message = format!("completed ({})", branch.termination.as_str());
                    }
                }
            }
            branches.push(branch);
        }
        let declined = declined.borrow().clone();
        Ok(RunResult { branches, meta, declined })
    }
}
