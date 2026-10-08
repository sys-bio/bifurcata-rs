//! Serialisation and the comparator (specification §10.3), ported check by
//! check from the Delphi `Bifurcata.Tests.Serialise`.
//!
//! The schema is a contract — console output, regression baselines and
//! persistence at once — so these pin full precision through a round trip, the
//! flat branches-with-parentId shape, and the comparator's rules.
//!
//! Differences from the Delphi suite: the round trip is asserted bit-exact
//! (Delphi's reader was not correctly rounded and could promise only one ULP);
//! the confidence mutation goes spectrum → coefficient, since this library has
//! no normal forms until M3; and mutations are made on the parsed JSON rather
//! than by replacing text.

use bifurcata::continuation::{ContinuationEngine, ContinuationOptions};
use bifurcata::equilibrium::EquilibriumCurve;
use bifurcata::serialise::*;
use bifurcata::test_problems::Brusselator;
use bifurcata::types::*;
use serde_json::Value;

/// A small but complete run: a Brusselator branch with its Hopf.
fn build_run() -> (Branch, RunMetadata, ContinuationOptions) {
    let options = ContinuationOptions { initial_step: 0.05, max_step: 0.1, max_points: 100, parameter_min: 1.0, parameter_max: 3.0, ..Default::default() };
    let curve = EquilibriumCurve::new(Brusselator, &[1.0, 1.2], 1, &options).unwrap();
    let x0 = curve.pack(&Brusselator::equilibrium(1.0, 1.2), 1.2);
    let mut engine = ContinuationEngine::new(curve, options.clone());
    engine.initialise(&x0, 1).unwrap();
    engine.run();
    let meta = RunMetadata {
        model_name: "brusselator".into(),
        model_source: "builtin:brusselator".into(),
        state_names: vec!["X".into(), "Y".into()],
        independent_count: 2,
        active_parameter_names: vec!["B".into()],
        active_parameter_indices: vec![1],
        range_lo: 1.0,
        range_hi: 3.0,
        direction: 1,
        exit_code: 1,
        message: "completed at parameterBound".into(),
    };
    (engine.into_branch(), meta, options)
}

fn to_string(branch: &Branch, meta: &RunMetadata, options: &ContinuationOptions) -> String {
    run_to_json_string(std::slice::from_ref(branch), meta, options, &SerialiseOptions::default())
}

#[test]
fn round_trip() {
    let (branch, meta, options) = build_run();
    assert!(branch.points.len() > 10, "the test run produced a branch");
    assert_eq!(branch.bifurcations.len(), 1, "the test run found the Hopf point");

    let s = to_string(&branch, &meta, &options);
    assert!(s.contains("\"schema\""), "the output carries a schema field");
    assert!(s.contains(SCHEMA_VERSION), "the schema version is {SCHEMA_VERSION}");
    assert!(s.contains("\"branches\""), "the output has a branches array");
    assert!(s.contains("\"bifurcations\""), "bifurcations are a FLAT list, not nested inside branches");
    assert!(s.contains("\"parentId\""), "branches carry parentId, so the tree needs no nested schema");
    assert!(!s.starts_with('\u{feff}'), "written without a byte-order mark");

    let loaded = load_run(&s).unwrap();
    assert_eq!(loaded.schema, SCHEMA_VERSION, "the schema round-trips");
    assert_eq!(loaded.branches.len(), 1, "one branch round-trips");
    assert_eq!(loaded.meta.model_name, "brusselator", "the model name round-trips");
    assert_eq!(loaded.meta.independent_count, 2, "the independent count round-trips");
    assert_eq!(loaded.meta.state_names, vec!["X", "Y"], "the state names round-trip, in order");
    assert_eq!(loaded.meta.exit_code, 1, "the exit code round-trips");

    let lb = &loaded.branches[0];
    assert_eq!(lb.points.len(), branch.points.len(), "the point count round-trips");
    assert_eq!(lb.termination, branch.termination, "the termination reason round-trips");

    // Bit-exact: the writer emits the shortest form that reads back to the same bits.
    for (p0, p1) in branch.points.iter().zip(&lb.points) {
        assert_eq!(p0.u, p1.u, "every state round-trips exactly");
        assert_eq!(p0.lambda[1], p1.lambda[0], "every active parameter round-trips exactly (only active ones are written)");
        assert_eq!(p0.s, p1.s, "every arclength round-trips exactly");
        assert_eq!(p0.eigenvalues, p1.eigenvalues, "the spectrum round-trips exactly");
        assert_eq!(p0.unstable_dim, p1.unstable_dim, "the unstable dimension round-trips");
    }

    assert_eq!(lb.bifurcations.len(), 1, "the bifurcation round-trips");
    let (ia, ib) = (&branch.bifurcations[0], &lb.bifurcations[0]);
    assert_eq!(ib.kind, ia.kind, "the bifurcation kind round-trips");
    assert_eq!(ib.confidence, ia.confidence, "the confidence round-trips");
    assert_eq!(ib.point.lambda[0], ia.point.lambda[1], "the bifurcation parameter round-trips exactly");
    assert_eq!(ib.normal_form.get("omega"), ia.normal_form.get("omega"), "the normal-form coefficient round-trips exactly");
}

#[test]
fn no_spectrum() {
    let (branch, meta, options) = build_run();
    let with = run_to_json_string(std::slice::from_ref(&branch), &meta, &options, &SerialiseOptions::default());
    let without = run_to_json_string(std::slice::from_ref(&branch), &meta, &options, &SerialiseOptions { include_spectrum: false, ..Default::default() });
    assert!(with.contains("\"eigenvalues\""), "the spectrum is present by default");
    assert!(!without.contains("\"eigenvalues\""), "--no-spectrum removes it entirely");
    assert!(without.len() < with.len(), "suppression shrinks the file ({} -> {} bytes)", with.len(), without.len());
    assert!(without.contains("\"u\""), "the points survive spectrum suppression");
}

/// The run, reloaded after `edit` has been applied to its JSON.
fn mutated(s: &str, edit: impl FnOnce(&mut Value)) -> LoadedRun {
    let mut v: Value = serde_json::from_str(s).unwrap();
    edit(&mut v);
    load_run(&v.to_string()).unwrap()
}

fn set_omega(s: &str, omega: f64) -> LoadedRun {
    mutated(s, |v| v["bifurcations"][0]["normalForm"]["omega"] = omega.into())
}

#[test]
fn comparator() {
    let (branch, meta, options) = build_run();
    let s = to_string(&branch, &meta, &options);
    let o = CompareOptions::default();
    let original = load_run(&s).unwrap();

    assert_eq!(compare_runs(&original, &load_run(&s).unwrap(), &o), Ok(()), "a run compares equal to itself");

    let why = compare_runs(&original, &mutated(&s, |v| v["bifurcations"][0]["kind"] = "Fold".into()), &o).expect_err("a changed kind is a mismatch");
    assert!(why.contains("kind differs"), "and the message names the kind difference: {why}");

    // §10.3: a changed confidence is a mismatch even at the same location.
    assert_eq!(branch.bifurcations[0].confidence, Confidence::Spectrum, "the run carries a spectrum-backed confidence to mutate");
    let why = compare_runs(&original, &mutated(&s, |v| v["bifurcations"][0]["confidence"] = "coefficient".into()), &o)
        .expect_err("a changed confidence is a mismatch even at the same location");
    assert!(why.contains("confidence differs"), "and the message names the confidence difference: {why}");

    let why = compare_runs(&original, &set_omega(&s, -branch.bifurcations[0].normal_form.get("omega").unwrap()), &o).expect_err("a flipped coefficient sign is a mismatch");
    assert!(why.contains("SIGN"), "and the message says the SIGN is wrong, not just the value: {why}");

    // A coefficient zero by definition (cusp, Bautin) comes out as noise of
    // either sign; two runs agreeing to eight places must compare equal.
    assert_eq!(compare_runs(&set_omega(&s, 5.8e-8), &set_omega(&s, -5.8e-8), &o), Ok(()), "+5.8e-8 against −5.8e-8 compares equal");
    // But only when BOTH are tiny.
    assert!(compare_runs(&set_omega(&s, 5.8e-8), &set_omega(&s, -0.5), &o).is_err(), "zero to a real negative value is still a mismatch");
    assert!(compare_runs(&set_omega(&s, 5.8e-4), &set_omega(&s, -5.8e-4), &o).is_err(), "a sign flip well above the zero tolerance is still a mismatch");

    let stopped = mutated(&s, |v| v["branches"][0]["terminationReason"] = "stepTooSmall".into());
    assert!(compare_runs(&original, &stopped, &o).is_err(), "a changed termination reason is a mismatch");

    // The curve itself: a point moved while the bifurcations stay put.
    let moved = mutated(&s, |v| {
        let y = v["branches"][0]["points"][5]["u"][1].as_f64().unwrap();
        v["branches"][0]["points"][5]["u"][1] = (y * 1.001).into();
    });
    let why = compare_runs(&original, &moved, &o).expect_err("a moved point is a mismatch");
    assert!(why.contains("point 5"), "and the message names the point: {why}");
    // The last point is always compared, whatever the stride.
    let last = branch.points.len() - 1;
    assert_ne!(last % 5, 0, "the test needs a last point off the stride");
    let moved_last = mutated(&s, |v| {
        let y = v["branches"][0]["points"][last]["u"][1].as_f64().unwrap();
        v["branches"][0]["points"][last]["u"][1] = (y * 1.001).into();
    });
    assert!(compare_runs(&original, &moved_last, &o).is_err(), "the last point is compared whatever the stride");

    // States are matched by name, so a reordering is not a difference...
    let reordered = mutated(&s, |v| {
        v["model"]["stateNames"] = serde_json::json!(["Y", "X"]);
        for p in v["branches"][0]["points"].as_array_mut().unwrap() {
            let u = p["u"].as_array().unwrap().clone();
            p["u"] = serde_json::json!([u[1], u[0]]);
        }
    });
    assert_eq!(compare_runs(&reordered, &original, &o), Ok(()), "the same states in another order compare equal");
    // ...but a renaming is.
    let renamed = mutated(&s, |v| v["model"]["stateNames"] = serde_json::json!(["X", "Z"]));
    assert!(compare_runs(&renamed, &original, &o).is_err(), "different state names are a mismatch");

    // Location-only comparison ignores what M2 cannot produce.
    let locations_only = CompareOptions { compare_normal_forms: false, ..o };
    let coefficient = mutated(&s, |v| {
        v["bifurcations"][0]["confidence"] = "coefficient".into();
        v["bifurcations"][0]["normalForm"]["l1"] = (-0.5).into();
    });
    assert_eq!(compare_runs(&coefficient, &original, &locations_only), Ok(()), "location-only ignores confidence and coefficients");
    let shifted = mutated(&s, |v| v["bifurcations"][0]["lambda"][0] = 2.001.into());
    assert!(compare_runs(&shifted, &original, &locations_only).is_err(), "but not the location");
}

#[test]
fn settings_file() {
    let mut o = ContinuationOptions::default();
    let unknown = apply_options_json(r#"{"initialStep": 0.25, "corrector": "moorePenrose", "predictor": "secant", "detectBifurcations": false}"#, &mut o).unwrap();
    assert_eq!(o.initial_step, 0.25, "initialStep is applied");
    assert_eq!(o.corrector, CorrectorKind::MoorePenrose, "the corrector is applied");
    assert_eq!(o.predictor, PredictorKind::Secant, "the predictor is applied");
    assert!(!o.detect_bifurcations, "a false boolean is applied");
    assert!(unknown.is_empty(), "no spurious unknown keys");

    // Absent keys are left alone, not reset.
    let mut o = ContinuationOptions { max_points: 777, ..Default::default() };
    apply_options_json(r#"{"initialStep": 0.1}"#, &mut o).unwrap();
    assert_eq!(o.max_points, 777, "a key absent from the file leaves its option untouched");

    // A typo is reported: ignoring it looks exactly like the option having no effect.
    let mut o = ContinuationOptions::default();
    let unknown = apply_options_json(r#"{"maxpoints": 10, "wibble": 3}"#, &mut o).unwrap();
    assert_eq!(unknown.len(), 2, "unrecognised keys are reported");
    assert!(unknown.contains(&"maxpoints".to_owned()), "a case-wrong key is reported rather than silently applied");
    assert_eq!(o.max_points, 1000, "and is not applied");

    // A whole run file works as a settings file.
    let mut o = ContinuationOptions::default();
    apply_options_json(r#"{"schema":"bifurcata/1","run":{"options":{"initialStep": 0.333}}}"#, &mut o).unwrap();
    assert_eq!(o.initial_step, 0.333, "a full run file is accepted as a settings file");

    // And our own output reproduces the options it was run with.
    let (branch, meta, options) = build_run();
    let mut o = ContinuationOptions::default();
    let unknown = apply_options_json(&to_string(&branch, &meta, &options), &mut o).unwrap();
    assert!(unknown.is_empty(), "every key written is understood: {unknown:?}");
    assert_eq!(o, options, "a run's output, read back as settings, gives the same options");
}

#[test]
fn csv() {
    let (branch, meta, _) = build_run();
    let s = run_to_csv(std::slice::from_ref(&branch), &meta);
    let lines: Vec<&str> = s.lines().collect();
    assert!(lines.len() > 10, "the CSV has rows");
    assert!(lines[0].starts_with("branch,point,s,X,Y,B"), "the header names the states and the active parameter: {}", lines[0]);
    assert!(!s.contains("kind"), "the CSV emits points only, no bifurcation records");
    assert_eq!(lines.len(), branch.points.len() + 1, "one row per point");
}

#[test]
fn delphi_files_load() {
    // A byte-order mark and Delphi's number spellings (1E-6) are accepted.
    let text = "\u{feff}{\"schema\":\"bifurcata/1\",\"run\":{\"activeParameters\":[\"B\"],\"range\":[1,4],\"direction\":0,\
                \"options\":{\"minStep\":1E-6}},\"branches\":[{\"id\":0,\"parentId\":null,\"originBifurcationId\":null,\
                \"points\":[{\"s\":0,\"u\":[1,1.2],\"lambda\":[1.2],\"unstableDim\":-1,\"stepSize\":0,\"newtonIterations\":0}],\
                \"terminationReason\":\"parameterBound\"}],\"bifurcations\":[],\"status\":{\"exitCode\":1,\"message\":\"\"}}";
    let run = load_run(text).expect("a Delphi file loads");
    assert_eq!(run.meta.direction, 0, "direction 0 (both) is read");
    assert_eq!(run.branches[0].points[0].unstable_dim, 0, "Delphi's −1 ('not known') reads as 0");
    let mut o = ContinuationOptions::default();
    apply_options_json(text, &mut o).unwrap();
    assert_eq!(o.min_step, 1e-6, "1E-6 is read");
}
