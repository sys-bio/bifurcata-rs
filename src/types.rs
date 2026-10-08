//! Shared types: solve status, bifurcation kinds, curve points and branches
//! (specification §4, §6, §9, §10.3).
//!
//! The string forms (`as_str`, `abbreviation`, …) are the **wire format** of
//! the `bifurcata/1` JSON schema (§10.3) and of the regression comparator. They
//! are part of the contract — changing one invalidates every stored baseline,
//! including the Delphi version's — so they are defined here, once.

use std::fmt;

use crate::complex::Complex;

// ---- Status reporting (§9.1) ------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveCode {
    Ok,
    /// Newton ran out of iterations.
    MaxIterations,
    /// The residual grew without bound.
    Diverged,
    /// A factorisation failed outright.
    SingularJacobian,
    /// The condition estimate is below the threshold.
    IllConditioned,
    /// Armijo could not find a decrease.
    LineSearchFailed,
    /// The continuation step fell below the minimum.
    StepTooSmall,
    /// A parameter or state bound was reached (not an error).
    BoundReached,
    InvalidInput,
    /// A linear-algebra routine failed (named `lapackError` on the wire, for
    /// compatibility with the Delphi version, which used LAPACK).
    LinAlgError,
    NotConverged,
}

impl SolveCode {
    pub fn as_str(self) -> &'static str {
        match self {
            SolveCode::Ok => "ok",
            SolveCode::MaxIterations => "maxIterations",
            SolveCode::Diverged => "diverged",
            SolveCode::SingularJacobian => "singularJacobian",
            SolveCode::IllConditioned => "illConditioned",
            SolveCode::LineSearchFailed => "lineSearchFailed",
            SolveCode::StepTooSmall => "stepTooSmall",
            SolveCode::BoundReached => "boundReached",
            SolveCode::InvalidInput => "invalidInput",
            SolveCode::LinAlgError => "lapackError",
            SolveCode::NotConverged => "notConverged",
        }
    }
}

/// Carried out of every failure path. `detail` is meant to be shown to a user
/// verbatim: continuation failures near folds are routine, and the user needs
/// to know which condition fired.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveStatus {
    pub code: SolveCode,
    pub detail: String,
    pub iterations: usize,
    pub residual_norm: f64,
}

impl SolveStatus {
    pub fn ok() -> Self {
        Self { code: SolveCode::Ok, detail: String::new(), iterations: 0, residual_norm: 0.0 }
    }

    pub fn ok_after(iterations: usize, residual_norm: f64) -> Self {
        Self { iterations, residual_norm, ..Self::ok() }
    }

    pub fn fail(code: SolveCode, detail: impl Into<String>) -> Self {
        Self { code, detail: detail.into(), iterations: 0, residual_norm: 0.0 }
    }

    pub fn is_ok(&self) -> bool {
        self.code == SolveCode::Ok
    }
}

impl fmt::Display for SolveStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_ok() {
            write!(f, "OK ({} iterations, residual {:.3e})", self.iterations, self.residual_norm)
        } else if self.detail.is_empty() {
            write!(f, "{}", self.code.as_str())
        } else {
            write!(f, "{}: {}", self.code.as_str(), self.detail)
        }
    }
}

impl std::error::Error for SolveStatus {}

// ---- Bifurcation classification ---------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BifurcationKind {
    None,
    // Codimension 1, equilibria
    /// LP — limit point / saddle-node
    Fold,
    /// BP — simple branch point
    BranchPoint,
    /// H
    Hopf,
    /// NS0 — a bialternate zero that is NOT a Hopf
    NeutralSaddle,
    // Codimension 2, equilibria
    /// CP
    Cusp,
    /// BT
    BogdanovTakens,
    /// ZH
    ZeroHopf,
    /// GH — generalised Hopf
    Bautin,
    /// HH — detected only
    HopfHopf,
    // Periodic orbits
    /// LPC
    FoldCycle,
    /// PD
    PeriodDoubling,
    /// NS
    NeimarkSacker,
    /// BPC
    BranchPointCycle,
    // Codimension 2, periodic orbits (detected only, §7.5)
    /// CPC — the fold-of-cycles coefficient vanishes
    CuspCycle,
    /// GPD — the period-doubling coefficient vanishes
    GeneralisedFlip,
    /// R1 — the angle reaches 0
    Resonance11,
    /// R2 — the angle reaches π
    Resonance12,
    /// R3 — the angle reaches 2π/3
    Resonance13,
    /// R4 — the angle reaches π/2
    Resonance14,
    /// LPPD
    FoldFlip,
    /// LPNS
    FoldNeimarkSacker,
    /// PDNS
    FlipNeimarkSacker,
    /// NSNS
    DoubleNeimarkSacker,
}

impl BifurcationKind {
    pub const ALL: [BifurcationKind; 24] = [
        BifurcationKind::None,
        BifurcationKind::Fold,
        BifurcationKind::BranchPoint,
        BifurcationKind::Hopf,
        BifurcationKind::NeutralSaddle,
        BifurcationKind::Cusp,
        BifurcationKind::BogdanovTakens,
        BifurcationKind::ZeroHopf,
        BifurcationKind::Bautin,
        BifurcationKind::HopfHopf,
        BifurcationKind::FoldCycle,
        BifurcationKind::PeriodDoubling,
        BifurcationKind::NeimarkSacker,
        BifurcationKind::BranchPointCycle,
        BifurcationKind::CuspCycle,
        BifurcationKind::GeneralisedFlip,
        BifurcationKind::Resonance11,
        BifurcationKind::Resonance12,
        BifurcationKind::Resonance13,
        BifurcationKind::Resonance14,
        BifurcationKind::FoldFlip,
        BifurcationKind::FoldNeimarkSacker,
        BifurcationKind::FlipNeimarkSacker,
        BifurcationKind::DoubleNeimarkSacker,
    ];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            BifurcationKind::None => "None",
            BifurcationKind::Fold => "Fold",
            BifurcationKind::BranchPoint => "BranchPoint",
            BifurcationKind::Hopf => "Hopf",
            BifurcationKind::NeutralSaddle => "NeutralSaddle",
            BifurcationKind::Cusp => "Cusp",
            BifurcationKind::BogdanovTakens => "BogdanovTakens",
            BifurcationKind::ZeroHopf => "ZeroHopf",
            BifurcationKind::Bautin => "Bautin",
            BifurcationKind::HopfHopf => "HopfHopf",
            BifurcationKind::FoldCycle => "FoldCycle",
            BifurcationKind::PeriodDoubling => "PeriodDoubling",
            BifurcationKind::NeimarkSacker => "NeimarkSacker",
            BifurcationKind::BranchPointCycle => "BranchPointCycle",
            BifurcationKind::CuspCycle => "CuspCycle",
            BifurcationKind::GeneralisedFlip => "GeneralisedFlip",
            BifurcationKind::Resonance11 => "Resonance1to1",
            BifurcationKind::Resonance12 => "Resonance1to2",
            BifurcationKind::Resonance13 => "Resonance1to3",
            BifurcationKind::Resonance14 => "Resonance1to4",
            BifurcationKind::FoldFlip => "FoldFlip",
            BifurcationKind::FoldNeimarkSacker => "FoldNeimarkSacker",
            BifurcationKind::FlipNeimarkSacker => "FlipNeimarkSacker",
            BifurcationKind::DoubleNeimarkSacker => "DoubleNeimarkSacker",
        }
    }

    /// From the wire name, ignoring case; `None` if unrecognised.
    pub fn parse(s: &str) -> BifurcationKind {
        Self::ALL
            .iter()
            .copied()
            .find(|k| k.as_str().eq_ignore_ascii_case(s))
            .unwrap_or(BifurcationKind::None)
    }

    /// The usual short label (LP, BP, H, …); empty for `None`.
    pub fn abbreviation(self) -> &'static str {
        match self {
            BifurcationKind::None => "",
            BifurcationKind::Fold => "LP",
            BifurcationKind::BranchPoint => "BP",
            BifurcationKind::Hopf => "H",
            BifurcationKind::NeutralSaddle => "NS0",
            BifurcationKind::Cusp => "CP",
            BifurcationKind::BogdanovTakens => "BT",
            BifurcationKind::ZeroHopf => "ZH",
            BifurcationKind::Bautin => "GH",
            BifurcationKind::HopfHopf => "HH",
            BifurcationKind::FoldCycle => "LPC",
            BifurcationKind::PeriodDoubling => "PD",
            BifurcationKind::NeimarkSacker => "NS",
            BifurcationKind::BranchPointCycle => "BPC",
            BifurcationKind::CuspCycle => "CPC",
            BifurcationKind::GeneralisedFlip => "GPD",
            BifurcationKind::Resonance11 => "R1",
            BifurcationKind::Resonance12 => "R2",
            BifurcationKind::Resonance13 => "R3",
            BifurcationKind::Resonance14 => "R4",
            BifurcationKind::FoldFlip => "LPPD",
            BifurcationKind::FoldNeimarkSacker => "LPNS",
            BifurcationKind::FlipNeimarkSacker => "PDNS",
            BifurcationKind::DoubleNeimarkSacker => "NSNS",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Classification {
    Unknown,
    Supercritical,
    Subcritical,
    /// The coefficient came out near zero; codimension 2 suspected.
    Degenerate,
}

impl Classification {
    pub fn as_str(self) -> &'static str {
        match self {
            Classification::Unknown => "unknown",
            Classification::Supercritical => "supercritical",
            Classification::Subcritical => "subcritical",
            Classification::Degenerate => "degenerate",
        }
    }
}

/// How a classification was reached. The comparator treats a change here as a
/// mismatch even when the location agrees (§10.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confidence {
    /// A normal-form coefficient decided it.
    Coefficient,
    /// Only an eigenvalue crossing was available.
    Spectrum,
    /// The coefficient was near zero.
    Degenerate,
}

impl Confidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Confidence::Coefficient => "coefficient",
            Confidence::Spectrum => "spectrum",
            Confidence::Degenerate => "degenerate",
        }
    }
}

// ---- Continuation stepping ---------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepResult {
    /// Point accepted; continue.
    Ok,
    /// Target reached.
    Converged,
    /// The corrector gave up.
    Failed,
    /// A parameter or state bound was reached.
    Boundary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationReason {
    Running,
    MaxPoints,
    ParameterBound,
    StateBound,
    StepTooSmall,
    CorrectorFailed,
    UserAborted,
    /// Returned to the starting point.
    ClosedCurve,
}

impl TerminationReason {
    pub const ALL: [TerminationReason; 8] = [
        TerminationReason::Running,
        TerminationReason::MaxPoints,
        TerminationReason::ParameterBound,
        TerminationReason::StateBound,
        TerminationReason::StepTooSmall,
        TerminationReason::CorrectorFailed,
        TerminationReason::UserAborted,
        TerminationReason::ClosedCurve,
    ];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            TerminationReason::Running => "running",
            TerminationReason::MaxPoints => "maxPoints",
            TerminationReason::ParameterBound => "parameterBound",
            TerminationReason::StateBound => "stateBound",
            TerminationReason::StepTooSmall => "stepTooSmall",
            TerminationReason::CorrectorFailed => "correctorFailed",
            TerminationReason::UserAborted => "userAborted",
            TerminationReason::ClosedCurve => "closedCurve",
        }
    }

    pub fn parse(s: &str) -> Option<TerminationReason> {
        Self::ALL.iter().copied().find(|t| t.as_str().eq_ignore_ascii_case(s))
    }

    /// The console harness's exit code (§10.1): hitting a bound is a normal
    /// outcome (1); giving up at a fold is not (2).
    pub fn exit_code(self) -> i32 {
        match self {
            TerminationReason::Running
            | TerminationReason::MaxPoints
            | TerminationReason::ClosedCurve
            | TerminationReason::UserAborted => 0,
            TerminationReason::ParameterBound | TerminationReason::StateBound => 1,
            TerminationReason::StepTooSmall | TerminationReason::CorrectorFailed => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictorKind {
    Secant,
    Tangent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectorKind {
    Palc,
    MoorePenrose,
}

// ---- Normal-form coefficients -------------------------------------------------

/// An open name → value collection in insertion order, because the
/// coefficient set differs per kind: `a` for a fold, `l1` and `omega` for a
/// Hopf, a whole set for a Bogdanov–Takens (§10.3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NormalForm {
    entries: Vec<(String, f64)>,
}

impl NormalForm {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Set a coefficient, replacing any existing value of the same name.
    pub fn set(&mut self, name: &str, value: f64) {
        match self.entries.iter_mut().find(|(n, _)| n == name) {
            Some(entry) => entry.1 = value,
            None => self.entries.push((name.to_owned(), value)),
        }
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.entries.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }

    pub fn get_or(&self, name: &str, default: f64) -> f64 {
        self.get(name).unwrap_or(default)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> {
        self.entries.iter().map(|(n, v)| (n.as_str(), *v))
    }
}

// ---- Points and bifurcation records -------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CurvePoint {
    /// Arclength along the branch.
    pub s: f64,
    /// The reduced state (independent species only).
    pub u: Vec<f64>,
    /// **Every** parameter, not just the active one: read it at the active index.
    pub lambda: Vec<f64>,
    /// The spectrum of F_u at this point.
    pub eigenvalues: Vec<Complex>,
    /// The number of eigenvalues with Re μ > 0.
    pub unstable_dim: usize,
    pub step_size: f64,
    pub newton_iterations: usize,
    /// Test-function values, for the trace file.
    pub test_functions: Vec<f64>,
    /// Periodic orbits only; 0 for an equilibrium.
    pub period: f64,
    /// The range of each state over a periodic orbit; empty for an
    /// equilibrium. A cycle's `u` is its state at t = 0, which moves with the
    /// phase condition; the range does not.
    pub u_min: Vec<f64>,
    pub u_max: Vec<f64>,
    /// The tangent in the defining system's own unknown vector. Set only on a
    /// point carried by a [`BifurcationInfo`]: branch switching needs to know
    /// which tangent we *arrived* on, which a decoded point cannot say.
    pub tangent: Vec<f64>,
    /// The point in the defining system's own unknown vector, undecoded. Set on
    /// the same points as `tangent`: anything that continues from a located
    /// bifurcation needs it.
    pub raw_x: Vec<f64>,
}

impl CurvePoint {
    pub fn new(state_dim: usize, param_dim: usize) -> Self {
        Self { u: vec![0.0; state_dim], lambda: vec![0.0; param_dim], ..Default::default() }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BifurcationInfo {
    pub id: usize,
    pub branch_id: usize,
    pub kind: BifurcationKind,
    pub point: CurvePoint,
    pub normal_form: NormalForm,
    pub classification: Classification,
    pub confidence: Confidence,
    /// Free text for the report.
    pub detail: String,
}

impl Default for BifurcationInfo {
    fn default() -> Self {
        Self {
            id: 0,
            branch_id: 0,
            kind: BifurcationKind::None,
            point: CurvePoint::default(),
            normal_form: NormalForm::new(),
            classification: Classification::Unknown,
            confidence: Confidence::Spectrum,
            detail: String::new(),
        }
    }
}

// ---- Branch ----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Branch {
    pub id: usize,
    /// `None` for a root branch.
    pub parent_id: Option<usize>,
    /// `None` if the branch did not start at a bifurcation.
    pub origin_bifurcation_id: Option<usize>,
    pub points: Vec<CurvePoint>,
    pub bifurcations: Vec<BifurcationInfo>,
    pub termination: TerminationReason,
    pub termination_detail: String,
    pub state_names: Vec<String>,
    pub parameter_names: Vec<String>,
}

impl Branch {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            parent_id: None,
            origin_bifurcation_id: None,
            points: Vec::new(),
            bifurcations: Vec::new(),
            termination: TerminationReason::Running,
            termination_detail: String::new(),
            state_names: Vec::new(),
            parameter_names: Vec::new(),
        }
    }

    pub fn add_point(&mut self, p: CurvePoint) {
        self.points.push(p);
    }

    pub fn add_bifurcation(&mut self, b: BifurcationInfo) {
        self.bifurcations.push(b);
    }

    pub fn last_point(&self) -> Option<&CurvePoint> {
        self.points.last()
    }

    /// The range covered by parameter `k`, or `None` for an empty branch.
    pub fn parameter_range(&self, k: usize) -> Option<(f64, f64)> {
        let mut values = self.points.iter().map(|p| p.lambda[k]);
        let first = values.next()?;
        Some(values.fold((first, first), |(lo, hi), v| (lo.min(v), hi.max(v))))
    }
}
