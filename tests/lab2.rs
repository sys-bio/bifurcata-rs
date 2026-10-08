//! MatCont's Lab 2 predator–prey model: the case that motivated
//! `refine_and_locate` (the Delphi CLAUDE.md, "Test functions are built on the
//! tangent").
//!
//! Its branch point has a closed form. On the prey-free branch x = 0 the
//! predator equation gives c y² + d y + c b² = 0, and the branch point is where
//! ∂ẋ/∂x = r − y/a vanishes, at y = ra:
//!
//! ```text
//! d = −c (r²a² + b²) / (ra) = −0.3130208333…
//! ```
//!
//! The coexistence branch reaches it with a state crossing zero fast, so a
//! zero located by bisection from the full maximum step comes out at −0.3094,
//! 1.2% out (the Delphi version's figure before refinement, reproduced here
//! with refinement switched off). Re-walking the bracket in real steps first
//! gets it to eight figures.
#![cfg(feature = "antimony")]

use bifurcata::antimony::AntimonyProblem;
use bifurcata::continuation::ContinuationOptions;
use bifurcata::run::EquilibriumRun;
use bifurcata::types::*;

const LAB2: &str = "
model lab2
    -> x; r*x*(1-x)-x*y /(x+a)
    -> y; -c*y+x*y /(x+a)-d*y^2/(y^2+b^2)
    r = 2
    a = 0.6
    b = 0.25
    c = 0.25
    d = 0.1
    x = 1.2
    y = 1
end";

#[test]
fn branch_point_located_to_eight_figures_at_the_default_step() {
    let problem = AntimonyProblem::parse(LAB2).unwrap();
    let lambda0 = problem.parameter_values();
    let u0 = problem.find_steady_state(&lambda0).unwrap();
    // The default maximum step (0.1): the hard case.
    let options = ContinuationOptions { parameter_min: -1.0, parameter_max: 1.0, ..Default::default() };
    let run = EquilibriumRun {
        problem: &problem,
        model_name: "lab2".into(),
        model_source: "Lab2_Matcont.ant".into(),
        lambda0,
        active: problem.parameter_index("d").unwrap(),
        u0,
        direction: 0,
        options,
    }
    .run()
    .unwrap();
    let bps: Vec<&BifurcationInfo> = run.branches.iter().flat_map(|b| &b.bifurcations).filter(|b| b.kind == BifurcationKind::BranchPoint).collect();
    assert_eq!(bps.len(), 1, "one branch point");
    let (r, a, b, c) = (2.0, 0.6, 0.25, 0.25);
    let exact = -c * (r * r * a * a + b * b) / (r * a);
    let d = bps[0].point.lambda[problem.parameter_index("d").unwrap()];
    assert!((d - exact).abs() < 1e-8, "branch point at d = {d:.10}, exact {exact:.10}");
    assert!(bps[0].point.u[0].abs() < 1e-6, "on the prey-free branch x = 0 (x = {:.3e})", bps[0].point.u[0]);
    assert!((bps[0].point.u[1] - r * a).abs() < 1e-6, "at y = ra");
}
