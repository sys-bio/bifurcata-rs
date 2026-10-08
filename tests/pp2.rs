//! PP2, AUTO-07p's predator–prey demo (`demos/pp2`), whose first three
//! equilibrium branches are solvable in closed form — a target known
//! independently of any discretisation (the Delphi CLAUDE.md, "PP2 is the
//! closed-form model"):
//!
//! ```text
//! trivial branch      BP  p1 = (p2/p3)            = 0.6
//!                     NS0 p1 = (p2 − 1)/p3        = 0.4  (a neutral saddle, NOT a Hopf)
//! fish-only branch    BP  p1 = (2/3)/(1 − e^(−5/3)) = 0.821904345374
//!                     LP                          = 0.832929322234
//!                     BP  p1 = p2/p3              = 0.6  (crossing the trivial branch)
//! coexistence branch  H   p1 = 0.671593847479, ω = 0.604782221942
//! ```
//!
//! The trivial branch's Jacobian is diag(p2 − p1 p3, −1); on the fish-only
//! branch sharks = 0 leaves a scalar equation; on the coexistence branch fish
//! is pinned at 1/p4. Each branch is started explicitly, as the Delphi
//! `regen.bat` did: switching between them is M3. The fish-only branch passes
//! through (1, 0) at p1 = 0.
#![cfg(feature = "antimony")]

use bifurcata::antimony::AntimonyProblem;
use bifurcata::continuation::ContinuationOptions;
use bifurcata::run::EquilibriumRun;
use bifurcata::types::*;

const PP2: &str = "
model pp2
    -> fish;    p2*fish*(1 - fish) - fish*sharks - p1*(1 - exp(-p3*fish));
    -> sharks;  -sharks + p4*fish*sharks;
    p1 = 0;
    p2 = 3;
    p3 = 5;
    p4 = 3;
    fish   = 0;
    sharks = 0;
end";

/// The bifurcations on the branches from `u0` at p1 = 0, both directions, with
/// the model's own [bifurcation] settings (ds 0.01, dsMax 0.02, 600 steps).
fn bifurcations_from(u0: &[f64]) -> Vec<BifurcationInfo> {
    let problem = AntimonyProblem::parse(PP2).unwrap();
    let options = ContinuationOptions { initial_step: 0.01, max_step: 0.02, max_points: 600, parameter_min: 0.0, parameter_max: 1.0, ..Default::default() };
    let run = EquilibriumRun {
        problem: &problem,
        model_name: "pp2".into(),
        model_source: "pp2.ant".into(),
        lambda0: problem.parameter_values(),
        active: problem.parameter_index("p1").unwrap(),
        u0: u0.to_vec(),
        direction: 0,
        options,
    }
    .run()
    .unwrap();
    run.branches.into_iter().flat_map(|b| b.bifurcations).collect()
}

fn located(bifurcations: &[BifurcationInfo], kind: BifurcationKind) -> Vec<f64> {
    bifurcations.iter().filter(|b| b.kind == kind).map(|b| b.point.lambda[0]).collect()
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!((actual - expected).abs() <= tol, "{what}: got {actual:.12}, expected {expected:.12} (tolerance {tol:e})");
}

#[test]
fn trivial_branch() {
    let b = bifurcations_from(&[0.0, 0.0]);
    assert_eq!(b.len(), 2, "two records on the trivial branch: {:?}", b.iter().map(|i| i.kind).collect::<Vec<_>>());
    let bp = located(&b, BifurcationKind::BranchPoint);
    assert_eq!(bp.len(), 1, "one branch point");
    close(bp[0], 0.6, 1e-8, "BP at p1 = p2/p3");
    // The bialternate discrimination on a case where the answer is known exactly.
    assert!(located(&b, BifurcationKind::Hopf).is_empty(), "the zero of ψ_H at 0.4 is not a Hopf");
    let ns = located(&b, BifurcationKind::NeutralSaddle);
    assert_eq!(ns.len(), 1, "one neutral saddle");
    close(ns[0], 0.4, 1e-8, "neutral saddle at p1 = (p2 − 1)/p3");
}

#[test]
fn fish_only_branch() {
    let b = bifurcations_from(&[1.0, 0.0]);
    // Two branch points: where the coexistence branch leaves (closed form), and
    // at the far end, where the fish-only branch crosses the trivial one at
    // p1 = p2/p3 = 0.6. The corrector fails on the trials nearest a branch
    // point, so the second is located less sharply (Delphi: 0.599998 and
    // 0.60000005). A trial landing on the trivial branch would put it nearer
    // 0.596.
    let mut bp = located(&b, BifurcationKind::BranchPoint);
    bp.sort_by(f64::total_cmp);
    assert_eq!(bp.len(), 2, "two branch points on the fish-only branch: {:?}", b.iter().map(|i| (i.kind, i.point.lambda[0])).collect::<Vec<_>>());
    close(bp[1], (2.0 / 3.0) / (1.0 - (-5.0f64 / 3.0).exp()), 1e-8, "BP at p1 = (2/3)/(1 − e^(−5/3))");
    close(bp[0], 0.6, 1e-5, "BP where the fish-only branch crosses the trivial one");
    let lp = located(&b, BifurcationKind::Fold);
    assert_eq!(lp.len(), 1, "one fold on the fish-only branch");
    close(lp[0], 0.832929322234, 1e-8, "the fold");
    assert!(located(&b, BifurcationKind::Hopf).is_empty(), "no Hopf on the fish-only branch");
}

#[test]
fn coexistence_branch() {
    let b = bifurcations_from(&[1.0 / 3.0, 2.0]);
    let hopf: Vec<&BifurcationInfo> = b.iter().filter(|i| i.kind == BifurcationKind::Hopf).collect();
    assert_eq!(hopf.len(), 1, "one Hopf on the coexistence branch: {:?}", b.iter().map(|i| (i.kind, i.point.lambda[0])).collect::<Vec<_>>());
    close(hopf[0].point.lambda[0], 0.671593847479, 1e-8, "the Hopf");
    // ω comes from the eigenvalues of a finite-difference Jacobian; it is
    // 1.0e-8 high, exactly as the Delphi baseline's 0.6047822324 is.
    close(hopf[0].normal_form.get_or("omega", 0.0), 0.604782221942, 1e-7, "ω at the Hopf");
    close(hopf[0].point.u[0], 1.0 / 3.0, 1e-9, "fish pinned at 1/p4 on the coexistence branch");
    let bp = located(&b, BifurcationKind::BranchPoint);
    assert_eq!(bp.len(), 1, "where it meets the fish-only branch");
    close(bp[0], 0.821904345374, 1e-8, "BP at the same p1 as from the fish-only side");
}
