//! Runs checked against the Delphi version's baselines, read in place from the
//! Delphi project (nothing there is changed). Skipped when that project is not
//! on this machine; set `BIFURCATA_DELPHI_DIR` to point elsewhere.
//!
//! Compared: branch and bifurcation counts, kinds, and locations to six
//! significant figures (plan §8's exit criterion), the termination reasons,
//! and the curves themselves (point counts, and every 5th point and the last
//! to 1e-6).
//! Not compared: confidence, classification and normal-form coefficients,
//! which are milestone M3.

use std::path::PathBuf;

use bifurcata::continuation::ContinuationOptions;
use bifurcata::models::{builtin_start_point, create_builtin};
use bifurcata::run::EquilibriumRun;
use bifurcata::serialise::*;

fn baselines_dir() -> Option<PathBuf> {
    let dir = std::env::var("BIFURCATA_DELPHI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(r"D:\Documents\Embarcadero\Studio\Projects\Bifurcation_Delphi"))
        .join("baselines");
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!("skipped: the Delphi baselines are not at {}", dir.display());
        None
    }
}

/// Run a built-in model as the baseline's `run` section describes, and return
/// our run reloaded from its JSON (so both sides carry active parameters only).
fn rerun_builtin(baseline: &LoadedRun, baseline_text: &str) -> LoadedRun {
    let name = &baseline.meta.model_name;
    let (problem, lambda0) = create_builtin(name).unwrap_or_else(|| panic!("{name} is not a built-in model"));
    let param = &baseline.meta.active_parameter_names[0];
    let active = (0..problem.parameter_count()).find(|&k| &problem.parameter_name(k) == param).unwrap();
    let mut options = ContinuationOptions::default();
    let unknown = apply_options_json(baseline_text, &mut options).unwrap();
    assert!(unknown.is_empty(), "{name}: unrecognised options {unknown:?}");
    let u0 = builtin_start_point(name, &lambda0).unwrap();
    let run = EquilibriumRun {
        problem: problem.as_ref(),
        model_name: name.clone(),
        model_source: format!("builtin:{name}"),
        lambda0,
        active,
        u0,
        direction: baseline.meta.direction,
        options: options.clone(),
    }
    .run()
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    load_run(&run_to_json_string(&run.branches, &run.meta, &options, &SerialiseOptions::default())).unwrap()
}

fn summary(run: &LoadedRun) -> String {
    run.branches
        .iter()
        .map(|b| {
            let bifs: Vec<String> = b.bifurcations.iter().map(|i| format!("{} {:.9}", i.kind.abbreviation(), i.point.lambda[0])).collect();
            format!("branch {}: {} points, {}, [{}]", b.id, b.points.len(), b.termination.as_str(), bifs.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n    ")
}

#[test]
fn builtin_models_match_the_delphi_baselines() {
    let Some(dir) = baselines_dir() else { return };
    let locations = CompareOptions { compare_normal_forms: false, compare_points: false, ..Default::default() };
    let mut failures = Vec::new();
    for name in ["brusselator", "selkov", "saddlenode", "transcritical"] {
        let text = std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap();
        let baseline = load_run(&text).unwrap();
        let ours = rerun_builtin(&baseline, &text);
        if let Err(why) = compare_runs(&ours, &baseline, &locations) {
            failures.push(format!("{name}: {why}\n    ours:     {}\n    baseline: {}", summary(&ours), summary(&baseline)));
        }
    }
    assert!(failures.is_empty(), "mismatches against the Delphi baselines:\n{}", failures.join("\n"));
}

/// The curve as well: every 5th point and the last, to the same tolerance.
/// Finite-difference details could in principle change the step sequence; on
/// these closed-form models they should not.
#[test]
fn builtin_model_points_match_the_delphi_baselines() {
    let Some(dir) = baselines_dir() else { return };
    let with_points = CompareOptions { compare_normal_forms: false, ..Default::default() };
    let mut failures = Vec::new();
    for name in ["brusselator", "selkov", "saddlenode", "transcritical"] {
        let text = std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap();
        let baseline = load_run(&text).unwrap();
        let ours = rerun_builtin(&baseline, &text);
        if let Err(why) = compare_runs(&ours, &baseline, &with_points) {
            failures.push(format!("{name}: {why}"));
        }
    }
    assert!(failures.is_empty(), "point mismatches against the Delphi baselines:\n{}", failures.join("\n"));
}

/// The Antimony baselines (`ant_*.json`, and `csum_edelstein.json`, whose
/// parameter is a conserved total): each run's model, parameter and options
/// from the baseline's own `run` section; the start from the steady-state
/// solver, or from the explicit `--start` the Delphi `regen.bat` used.
/// `ant_pp2_switch` needs branch switching, which is M3.
#[cfg(feature = "antimony")]
#[test]
fn antimony_models_match_the_delphi_baselines() {
    use bifurcata::antimony::AntimonyProblem;

    let Some(dir) = baselines_dir() else { return };
    let models = dir.parent().unwrap().join("GUIApp").join("Win64").join("Debug").join("models");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|f| (f.starts_with("ant_") || f == "csum_edelstein.json") && f != "ant_pp2_switch.json")
        .collect();
    files.sort();
    assert!(files.len() >= 20, "expected the Antimony baselines, found {}", files.len());

    let locations = CompareOptions { compare_normal_forms: false, ..Default::default() };
    let mut report = Vec::new();
    let mut failures = 0;
    for file in &files {
        let text = std::fs::read_to_string(dir.join(file)).unwrap();
        let baseline = load_run(&text).unwrap();
        let name = baseline.meta.model_name.clone();
        let problem = AntimonyProblem::parse(&std::fs::read_to_string(models.join(format!("{name}.ant"))).unwrap()).unwrap();
        let param = &baseline.meta.active_parameter_names[0];
        let active = problem.parameter_index(param).unwrap_or_else(|| panic!("{file}: no parameter {param}"));
        let lambda0 = problem.parameter_values();
        let start: Option<Vec<f64>> = match file.as_str() {
            "ant_pp2_trivial.json" => Some(vec![0.0, 0.0]),
            "ant_pp2_coexist.json" => Some(vec![0.33333333333333, 2.0]),
            _ => None,
        };
        let u0 = match start {
            Some(u) => u,
            None => match problem.find_steady_state(&lambda0) {
                Ok(u) => u,
                Err(e) => {
                    failures += 1;
                    report.push(format!("{file}: no steady state: {e}"));
                    continue;
                }
            },
        };
        let mut options = ContinuationOptions::default();
        apply_options_json(&text, &mut options).unwrap();
        let run = EquilibriumRun {
            problem: &problem,
            model_name: name.clone(),
            model_source: format!("{name}.ant"),
            lambda0,
            active,
            u0,
            direction: baseline.meta.direction,
            options: options.clone(),
        }
        .run();
        let ours = match run {
            Ok(r) => load_run(&run_to_json_string(&r.branches, &r.meta, &options, &SerialiseOptions::default())).unwrap(),
            Err(e) => {
                failures += 1;
                report.push(format!("{file}: {e}"));
                continue;
            }
        };
        match compare_runs(&ours, &baseline, &locations) {
            Ok(()) => report.push(format!("{file}: ok ({} bifurcations)", baseline.total_bifurcations())),
            Err(why) => {
                failures += 1;
                report.push(format!("{file}: {why}\n    ours:     {}\n    baseline: {}", summary(&ours), summary(&baseline)));
            }
        }
    }
    eprintln!("{}", report.join("\n"));
    assert_eq!(failures, 0, "Antimony baselines that do not match:\n{}", report.join("\n"));
}
