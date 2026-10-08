//! The `bifurcata/1` output schema, settings files, CSV, and the regression
//! comparator (specification §10.1, §10.3).
//!
//! One schema serves console output, regression baselines and persistence, so
//! it is a contract: the field names and the wire names in [`crate::types`]
//! are those of the Delphi version, and its baselines load here unchanged.
//!
//! As in the Delphi writer, `lambda` is always an **array** holding only the
//! active parameters (named in `run.activeParameters`): a codim-2 curve has two,
//! and a model may carry hundreds of parameters of which one varies. A loaded
//! point's `lambda` therefore holds the active parameters only.
//!
//! Numbers are written in the shortest form that reads back to the same bits,
//! so a round trip is exact (the Delphi version could only promise one ULP).
//! NaN and the infinities, which JSON lacks, are written as the strings
//! `"NaN"`, `"Infinity"` and `"-Infinity"`.

use serde_json::{Map, Value, json};

use crate::complex::{Complex, cx};
use crate::continuation::ContinuationOptions;
use crate::types::*;

pub const SCHEMA_VERSION: &str = "bifurcata/1";

/// Everything about a run that a [`Branch`] does not carry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunMetadata {
    pub model_name: String,
    pub model_source: String,
    pub state_names: Vec<String>,
    pub independent_count: usize,
    pub active_parameter_names: Vec<String>,
    /// Indices into each point's full `lambda`. After loading, `0..k`: a loaded
    /// point holds the active parameters only.
    pub active_parameter_indices: Vec<usize>,
    pub range_lo: f64,
    pub range_hi: f64,
    /// +1, −1, or 0 for both directions.
    pub direction: i32,
    pub exit_code: i32,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SerialiseOptions {
    /// `--no-spectrum` turns this off.
    pub include_spectrum: bool,
    pub include_test_functions: bool,
    pub indent: bool,
}

impl Default for SerialiseOptions {
    fn default() -> Self {
        Self { include_spectrum: true, include_test_functions: false, indent: true }
    }
}

// ---- Writing --------------------------------------------------------------------

fn num(v: f64) -> Value {
    if v.is_nan() {
        Value::from("NaN")
    } else if v.is_infinite() {
        Value::from(if v > 0.0 { "Infinity" } else { "-Infinity" })
    } else {
        Value::from(v)
    }
}

fn nums(v: &[f64]) -> Value {
    Value::Array(v.iter().map(|x| num(*x)).collect())
}

fn eigenvalues_json(v: &[Complex]) -> Value {
    Value::Array(v.iter().map(|z| json!({ "re": num(z.re), "im": num(z.im) })).collect())
}

/// Only the active parameters (see the module note); the whole vector when no
/// active set is declared, so the file is never silently empty.
fn active_lambda(lambda: &[f64], indices: &[usize]) -> Value {
    if indices.is_empty() {
        return nums(lambda);
    }
    Value::Array(indices.iter().filter(|&&k| k < lambda.len()).map(|&k| num(lambda[k])).collect())
}

fn predictor_name(p: PredictorKind) -> &'static str {
    match p {
        PredictorKind::Tangent => "tangent",
        PredictorKind::Secant => "secant",
    }
}

fn corrector_name(c: CorrectorKind) -> &'static str {
    match c {
        CorrectorKind::Palc => "palc",
        CorrectorKind::MoorePenrose => "moorePenrose",
    }
}

pub fn options_to_json(o: &ContinuationOptions) -> Value {
    json!({
        "initialStep": num(o.initial_step),
        "minStep": num(o.min_step),
        "maxStep": num(o.max_step),
        "stepIncreaseFactor": num(o.step_increase_factor),
        "stepDecreaseFactor": num(o.step_decrease_factor),
        "targetNewtonIterations": o.target_newton_iterations,
        "maxNewtonIterations": o.max_newton_iterations,
        "tolResidual": num(o.tol_residual),
        "tolStep": num(o.tol_step),
        "useLineSearch": o.use_line_search,
        "maxPoints": o.max_points,
        "parameterMin": num(o.parameter_min),
        "parameterMax": num(o.parameter_max),
        "stateBoundMax": num(o.state_bound_max),
        "detectBifurcations": o.detect_bifurcations,
        "computeNormalForms": o.compute_normal_forms,
        "bisectionTolerance": num(o.bisection_tolerance),
        "maxBisections": o.max_bisections,
        "eigenvalueZeroTol": num(o.eigenvalue_zero_tol),
        "maxTangentAngleDeg": num(o.max_tangent_angle_deg),
        "predictor": predictor_name(o.predictor),
        "corrector": corrector_name(o.corrector),
    })
}

fn point_json(p: &CurvePoint, active: &[usize], so: &SerialiseOptions) -> Value {
    let mut o = Map::new();
    o.insert("s".into(), num(p.s));
    o.insert("u".into(), nums(&p.u));
    o.insert("lambda".into(), active_lambda(&p.lambda, active));
    o.insert("unstableDim".into(), p.unstable_dim.into());
    if so.include_spectrum {
        o.insert("eigenvalues".into(), eigenvalues_json(&p.eigenvalues));
    }
    o.insert("stepSize".into(), num(p.step_size));
    o.insert("newtonIterations".into(), p.newton_iterations.into());
    if p.period != 0.0 {
        o.insert("period".into(), num(p.period));
    }
    // A periodic orbit's range: its u alone does not describe the solution.
    if !p.u_min.is_empty() {
        o.insert("uMin".into(), nums(&p.u_min));
    }
    if !p.u_max.is_empty() {
        o.insert("uMax".into(), nums(&p.u_max));
    }
    if so.include_test_functions && !p.test_functions.is_empty() {
        o.insert("testFunctions".into(), nums(&p.test_functions));
    }
    Value::Object(o)
}

fn optional_id(id: Option<usize>) -> Value {
    id.map_or(Value::Null, Value::from)
}

pub fn run_to_json(branches: &[Branch], meta: &RunMetadata, options: &ContinuationOptions, so: &SerialiseOptions) -> Value {
    let active = &meta.active_parameter_indices;
    let mut branch_values = Vec::new();
    let mut bifurcations = Vec::new();
    for b in branches {
        let mut bo = Map::new();
        bo.insert("id".into(), b.id.into());
        bo.insert("parentId".into(), optional_id(b.parent_id));
        bo.insert("originBifurcationId".into(), optional_id(b.origin_bifurcation_id));
        bo.insert("points".into(), Value::Array(b.points.iter().map(|p| point_json(p, active, so)).collect()));
        bo.insert("terminationReason".into(), b.termination.as_str().into());
        if !b.termination_detail.is_empty() {
            bo.insert("terminationDetail".into(), b.termination_detail.clone().into());
        }
        branch_values.push(Value::Object(bo));

        for info in &b.bifurcations {
            let mut io = Map::new();
            io.insert("id".into(), info.id.into());
            io.insert("branchId".into(), b.id.into());
            io.insert("kind".into(), info.kind.as_str().into());
            io.insert("s".into(), num(info.point.s));
            io.insert("u".into(), nums(&info.point.u));
            io.insert("lambda".into(), active_lambda(&info.point.lambda, active));
            if so.include_spectrum {
                io.insert("eigenvalues".into(), eigenvalues_json(&info.point.eigenvalues));
            }
            io.insert("normalForm".into(), Value::Object(info.normal_form.iter().map(|(k, v)| (k.to_owned(), num(v))).collect()));
            io.insert("classification".into(), info.classification.as_str().into());
            io.insert("confidence".into(), info.confidence.as_str().into());
            if !info.detail.is_empty() {
                io.insert("detail".into(), info.detail.clone().into());
            }
            bifurcations.push(Value::Object(io));
        }
    }
    json!({
        "schema": SCHEMA_VERSION,
        "model": {
            "name": meta.model_name,
            "source": meta.model_source,
            "stateNames": meta.state_names,
            "independentCount": meta.independent_count,
        },
        "run": {
            "activeParameters": meta.active_parameter_names,
            "range": [num(meta.range_lo), num(meta.range_hi)],
            "direction": meta.direction,
            "options": options_to_json(options),
        },
        "branches": branch_values,
        "bifurcations": bifurcations,
        "status": { "exitCode": meta.exit_code, "message": meta.message },
    })
}

/// The run as `bifurcata/1` JSON text, without a byte-order mark.
pub fn run_to_json_string(branches: &[Branch], meta: &RunMetadata, options: &ContinuationOptions, so: &SerialiseOptions) -> String {
    let v = run_to_json(branches, meta, options, so);
    let text = if so.indent { serde_json::to_string_pretty(&v) } else { serde_json::to_string(&v) };
    text.expect("a JSON value always serialises")
}

/// Points only, no bifurcation records (§10.1).
pub fn run_to_csv(branches: &[Branch], meta: &RunMetadata) -> String {
    let mut out = String::from("branch,point,s");
    for name in meta.state_names.iter().chain(&meta.active_parameter_names) {
        out.push(',');
        out.push_str(name);
    }
    out.push_str(",unstableDim,stepSize,newtonIterations\n");
    for b in branches {
        for (i, p) in b.points.iter().enumerate() {
            out.push_str(&format!("{},{},{}", b.id, i, p.s));
            for v in &p.u {
                out.push_str(&format!(",{v}"));
            }
            for &k in &meta.active_parameter_indices {
                out.push_str(&format!(",{}", p.lambda[k]));
            }
            out.push_str(&format!(",{},{},{}\n", p.unstable_dim, p.step_size, p.newton_iterations));
        }
    }
    out
}

// ---- Reading --------------------------------------------------------------------

/// A parsed run, for the comparator and for reloading.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadedRun {
    pub schema: String,
    pub meta: RunMetadata,
    pub branches: Vec<Branch>,
}

impl LoadedRun {
    pub fn total_bifurcations(&self) -> usize {
        self.branches.iter().map(|b| b.bifurcations.len()).sum()
    }
}

fn get_num(v: Option<&Value>, default: f64) -> f64 {
    match v {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::String(s)) => match s.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            _ => default,
        },
        _ => default,
    }
}

fn get_int(v: Option<&Value>, default: i64) -> i64 {
    match v {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).unwrap_or(default),
        _ => default,
    }
}

fn get_str(v: Option<&Value>) -> String {
    v.and_then(Value::as_str).unwrap_or_default().to_owned()
}

fn get_nums(v: Option<&Value>) -> Vec<f64> {
    v.and_then(Value::as_array).map(|a| a.iter().map(|x| get_num(Some(x), 0.0)).collect()).unwrap_or_default()
}

fn get_strs(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array).map(|a| a.iter().map(|x| x.as_str().unwrap_or_default().to_owned()).collect()).unwrap_or_default()
}

fn get_eigenvalues(v: Option<&Value>) -> Vec<Complex> {
    v.and_then(Value::as_array).map(|a| a.iter().map(|z| cx(get_num(z.get("re"), 0.0), get_num(z.get("im"), 0.0))).collect()).unwrap_or_default()
}

fn get_id(v: Option<&Value>) -> Option<usize> {
    v.and_then(Value::as_u64).map(|id| id as usize)
}

fn load_point(o: &Value) -> CurvePoint {
    CurvePoint {
        s: get_num(o.get("s"), 0.0),
        u: get_nums(o.get("u")),
        lambda: get_nums(o.get("lambda")),
        eigenvalues: get_eigenvalues(o.get("eigenvalues")),
        // The Delphi version writes −1 for "not known".
        unstable_dim: get_int(o.get("unstableDim"), 0).max(0) as usize,
        step_size: get_num(o.get("stepSize"), 0.0),
        newton_iterations: get_int(o.get("newtonIterations"), 0).max(0) as usize,
        test_functions: get_nums(o.get("testFunctions")),
        period: get_num(o.get("period"), 0.0),
        u_min: get_nums(o.get("uMin")),
        u_max: get_nums(o.get("uMax")),
        ..Default::default()
    }
}

fn parse_json(text: &str) -> Result<Value, String> {
    // Files written by the Delphi version start with a UTF-8 byte-order mark.
    serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| format!("not valid JSON: {e}"))
}

/// Parse a `bifurcata/1` run (a byte-order mark is skipped).
pub fn load_run(text: &str) -> Result<LoadedRun, String> {
    let root = parse_json(text)?;
    if !root.is_object() {
        return Err("a run must be a JSON object".to_owned());
    }
    let mut run = LoadedRun { schema: get_str(root.get("schema")), ..Default::default() };
    if let Some(model) = root.get("model") {
        run.meta.model_name = get_str(model.get("name"));
        run.meta.model_source = get_str(model.get("source"));
        run.meta.state_names = get_strs(model.get("stateNames"));
        run.meta.independent_count = get_int(model.get("independentCount"), 0).max(0) as usize;
    }
    if let Some(r) = root.get("run") {
        run.meta.active_parameter_names = get_strs(r.get("activeParameters"));
        let range = get_nums(r.get("range"));
        if range.len() >= 2 {
            run.meta.range_lo = range[0];
            run.meta.range_hi = range[1];
        }
        run.meta.direction = get_int(r.get("direction"), 1) as i32;
    }
    run.meta.active_parameter_indices = (0..run.meta.active_parameter_names.len()).collect();

    for (i, bo) in root.get("branches").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let mut b = Branch::new(get_id(bo.get("id")).unwrap_or(i));
        b.parent_id = get_id(bo.get("parentId"));
        b.origin_bifurcation_id = get_id(bo.get("originBifurcationId"));
        b.points = bo.get("points").and_then(Value::as_array).into_iter().flatten().map(load_point).collect();
        let reason = get_str(bo.get("terminationReason"));
        b.termination = TerminationReason::parse(&reason).ok_or_else(|| format!("branch {}: unknown termination reason \"{reason}\"", b.id))?;
        b.termination_detail = get_str(bo.get("terminationDetail"));
        b.state_names = run.meta.state_names.clone();
        b.parameter_names = run.meta.active_parameter_names.clone();
        run.branches.push(b);
    }

    // Bifurcations are stored flat with a branchId, so they attach afterwards.
    for (i, io) in root.get("bifurcations").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let mut info = BifurcationInfo {
            id: get_id(io.get("id")).unwrap_or(i),
            branch_id: get_id(io.get("branchId")).unwrap_or(0),
            kind: BifurcationKind::parse(&get_str(io.get("kind"))),
            point: CurvePoint {
                s: get_num(io.get("s"), 0.0),
                u: get_nums(io.get("u")),
                lambda: get_nums(io.get("lambda")),
                eigenvalues: get_eigenvalues(io.get("eigenvalues")),
                ..Default::default()
            },
            detail: get_str(io.get("detail")),
            ..Default::default()
        };
        if let Some(nf) = io.get("normalForm").and_then(Value::as_object) {
            for (name, v) in nf {
                if v.is_number() {
                    info.normal_form.set(name, get_num(Some(v), 0.0));
                }
            }
        }
        info.classification = match get_str(io.get("classification")).as_str() {
            "supercritical" => Classification::Supercritical,
            "subcritical" => Classification::Subcritical,
            "degenerate" => Classification::Degenerate,
            _ => Classification::Unknown,
        };
        info.confidence = match get_str(io.get("confidence")).as_str() {
            "coefficient" => Confidence::Coefficient,
            "degenerate" => Confidence::Degenerate,
            _ => Confidence::Spectrum,
        };
        if let Some(b) = run.branches.iter_mut().find(|b| b.id == info.branch_id) {
            b.add_bifurcation(info);
        }
    }

    if let Some(status) = root.get("status") {
        run.meta.exit_code = get_int(status.get("exitCode"), 0) as i32;
        run.meta.message = get_str(status.get("message"));
    }
    Ok(run)
}

/// The names [`apply_options_json`] understands, as [`options_to_json`] writes them.
const OPTION_KEYS: [&str; 22] = [
    "initialStep",
    "minStep",
    "maxStep",
    "stepIncreaseFactor",
    "stepDecreaseFactor",
    "targetNewtonIterations",
    "maxNewtonIterations",
    "tolResidual",
    "tolStep",
    "useLineSearch",
    "maxPoints",
    "parameterMin",
    "parameterMax",
    "stateBoundMax",
    "detectBifurcations",
    "computeNormalForms",
    "bisectionTolerance",
    "maxBisections",
    "eigenvalueZeroTol",
    "maxTangentAngleDeg",
    "predictor",
    "corrector",
];

/// Apply a JSON options record onto `options`, leaving absent keys alone, and
/// return the keys it did not recognise — a typo in a settings file is
/// otherwise a silent no-op that looks like the option had no effect. Accepts
/// a bare options object or a whole run file (options under `run.options`), so
/// a previous run's output can be reused as settings.
pub fn apply_options_json(text: &str, options: &mut ContinuationOptions) -> Result<Vec<String>, String> {
    let root = parse_json(text)?;
    let o = root.get("run").and_then(|r| r.get("options")).unwrap_or(&root);
    let o = o.as_object().ok_or("settings must be a JSON object")?;
    let f = |key: &str, target: &mut f64| {
        if let Some(v) = o.get(key) {
            *target = get_num(Some(v), *target);
        }
    };
    let n = |key: &str, target: &mut usize| {
        if let Some(v) = o.get(key) {
            *target = get_int(Some(v), *target as i64).max(0) as usize;
        }
    };
    let b = |key: &str, target: &mut bool| {
        if let Some(v) = o.get(key).and_then(Value::as_bool) {
            *target = v;
        }
    };
    f("initialStep", &mut options.initial_step);
    f("minStep", &mut options.min_step);
    f("maxStep", &mut options.max_step);
    f("stepIncreaseFactor", &mut options.step_increase_factor);
    f("stepDecreaseFactor", &mut options.step_decrease_factor);
    n("targetNewtonIterations", &mut options.target_newton_iterations);
    n("maxNewtonIterations", &mut options.max_newton_iterations);
    f("tolResidual", &mut options.tol_residual);
    f("tolStep", &mut options.tol_step);
    b("useLineSearch", &mut options.use_line_search);
    n("maxPoints", &mut options.max_points);
    f("parameterMin", &mut options.parameter_min);
    f("parameterMax", &mut options.parameter_max);
    f("stateBoundMax", &mut options.state_bound_max);
    b("detectBifurcations", &mut options.detect_bifurcations);
    b("computeNormalForms", &mut options.compute_normal_forms);
    f("bisectionTolerance", &mut options.bisection_tolerance);
    n("maxBisections", &mut options.max_bisections);
    f("eigenvalueZeroTol", &mut options.eigenvalue_zero_tol);
    f("maxTangentAngleDeg", &mut options.max_tangent_angle_deg);
    if let Some(p) = o.get("predictor").and_then(Value::as_str) {
        options.predictor = if p.eq_ignore_ascii_case("secant") { PredictorKind::Secant } else { PredictorKind::Tangent };
    }
    if let Some(c) = o.get("corrector").and_then(Value::as_str) {
        options.corrector = if c.eq_ignore_ascii_case("moorePenrose") { CorrectorKind::MoorePenrose } else { CorrectorKind::Palc };
    }
    Ok(o.keys().filter(|k| !OPTION_KEYS.contains(&k.as_str())).cloned().collect())
}

// ---- Comparing (§10.1, the `compare` command) -------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompareOptions {
    /// On parameter values (and on points), relative above 1.
    pub tolerance: f64,
    /// Looser, for normal-form coefficients; their signs must still match.
    pub coefficient_tolerance: f64,
    /// Below this magnitude a coefficient carries no sign. At a cusp or a
    /// Bautin point a coefficient is zero by definition, and what comes out is
    /// noise of either sign around 1e-8; two runs of the same code then differ
    /// in sign while agreeing to eight places. The sign is compared only where
    /// there is one — when either value exceeds this.
    pub coefficient_zero_tolerance: f64,
    /// Compare the curve as well as its bifurcations: otherwise a change that
    /// moves every point while leaving the bifurcations alone passes, and a run
    /// with no bifurcations is compared on its branch count alone.
    pub compare_points: bool,
    /// Compare every Nth point (and always the last).
    pub point_stride: usize,
    /// Compare confidence, classification and normal-form coefficients. Off
    /// for a location-only comparison: this library has no normal forms until
    /// M3, so against a Delphi baseline only kinds, counts and locations can
    /// agree.
    pub compare_normal_forms: bool,
}

impl Default for CompareOptions {
    fn default() -> Self {
        Self { tolerance: 1e-6, coefficient_tolerance: 1e-3, coefficient_zero_tolerance: 1e-6, compare_points: true, point_stride: 5, compare_normal_forms: true }
    }
}

fn describe(info: &BifurcationInfo) -> String {
    match info.point.lambda.first() {
        Some(l) => format!("{} at lambda = {l:.10}", info.kind.as_str()),
        None => format!("{} at lambda = ?", info.kind.as_str()),
    }
}

fn differs(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() > tol * b.abs().max(1.0)
}

/// Elementwise; both may be empty (an equilibrium has no orbit range), but a
/// change of length is a difference.
fn same_sampled(a: &[f64], b: &[f64], tol: f64, what: &str) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{what} count differs, got {}, baseline has {}", a.len(), b.len()));
    }
    match (0..b.len()).find(|&k| differs(a[k], b[k], tol)) {
        Some(k) => Err(format!("{what} {k} is {:.12}, baseline has {:.12} (tolerance {:.3e})", a[k], b[k], tol)),
        None => Ok(()),
    }
}

fn sign(v: f64) -> i32 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

/// Compare a run with a baseline: bifurcations by kind and location, then (as
/// configured) their classification and coefficients, the termination reason
/// and the curve itself. `Err` carries the **first** discrepancy.
///
/// States are matched **by name** when both runs name them: libRoadRunner
/// orders the independent species its own way, websim in model order, and the
/// same equilibrium must compare equal under either.
pub fn compare_runs(actual: &LoadedRun, baseline: &LoadedRun, o: &CompareOptions) -> Result<(), String> {
    // order[k] = the index in `actual` of the baseline's state k.
    let (names_a, names_b) = (&actual.meta.state_names, &baseline.meta.state_names);
    let order: Option<Vec<usize>> = if !names_a.is_empty() && names_a.len() == names_b.len() {
        let order: Option<Vec<usize>> = names_b.iter().map(|n| names_a.iter().position(|m| m == n)).collect();
        Some(order.ok_or_else(|| format!("state names differ: got {names_a:?}, baseline has {names_b:?}"))?)
    } else {
        None
    };
    let in_baseline_order = |u: &[f64]| -> Vec<f64> {
        match &order {
            Some(order) if u.len() == order.len() => order.iter().map(|&k| u[k]).collect(),
            _ => u.to_vec(),
        }
    };
    if actual.branches.len() != baseline.branches.len() {
        return Err(format!("branch count differs: got {}, baseline has {}", actual.branches.len(), baseline.branches.len()));
    }
    for (ba, bb) in actual.branches.iter().zip(&baseline.branches) {
        let id = bb.id;
        if ba.bifurcations.len() != bb.bifurcations.len() {
            return Err(format!("branch {id}: bifurcation count differs, got {}, baseline has {}", ba.bifurcations.len(), bb.bifurcations.len()));
        }
        for (i, (ia, ib)) in ba.bifurcations.iter().zip(&bb.bifurcations).enumerate() {
            if ia.kind != ib.kind {
                return Err(format!("branch {id}, bifurcation {i}: kind differs, got {}, baseline has {}", ia.kind.as_str(), ib.kind.as_str()));
            }
            let what = describe(ib);
            if ia.point.lambda.len() != ib.point.lambda.len() {
                return Err(format!("branch {id}, {what}: active parameter count differs"));
            }
            for (k, (a, b)) in ia.point.lambda.iter().zip(&ib.point.lambda).enumerate() {
                if differs(*a, *b, o.tolerance) {
                    return Err(format!("branch {id}, {what}: parameter {k} is {a:.12}, baseline has {b:.12} (tolerance {:.3e})", o.tolerance));
                }
            }
            if !o.compare_normal_forms {
                continue;
            }
            // §10.3: a change of confidence is a mismatch even at the same location.
            if ia.confidence != ib.confidence {
                return Err(format!("branch {id}, {what}: confidence differs, got {}, baseline has {}", ia.confidence.as_str(), ib.confidence.as_str()));
            }
            if ia.classification != ib.classification {
                return Err(format!(
                    "branch {id}, {what}: classification differs, got {}, baseline has {}",
                    ia.classification.as_str(),
                    ib.classification.as_str()
                ));
            }
            for (name, vb) in ib.normal_form.iter() {
                let Some(va) = ia.normal_form.get(name) else {
                    return Err(format!("branch {id}, {what}: normal-form coefficient \"{name}\" is missing"));
                };
                // The sign decides sub- from supercritical, so it must match —
                // unless both values are zero to within noise.
                if sign(va) != sign(vb) && (va.abs() > o.coefficient_zero_tolerance || vb.abs() > o.coefficient_zero_tolerance) {
                    return Err(format!("branch {id}, {what}: coefficient \"{name}\" has the wrong SIGN, got {va:.12}, baseline has {vb:.12}"));
                }
                if differs(va, vb, o.coefficient_tolerance) {
                    return Err(format!(
                        "branch {id}, {what}: coefficient \"{name}\" is {va:.12}, baseline has {vb:.12} (tolerance {:.3e})",
                        o.coefficient_tolerance
                    ));
                }
            }
        }
        if ba.termination != bb.termination {
            return Err(format!("branch {id}: termination differs, got {}, baseline has {}", ba.termination.as_str(), bb.termination.as_str()));
        }
        if o.compare_points {
            if ba.points.len() != bb.points.len() {
                return Err(format!("branch {id}: point count differs, got {}, baseline has {}", ba.points.len(), bb.points.len()));
            }
            let stride = o.point_stride.max(1);
            let mut i = 0;
            while i < bb.points.len() {
                let (pa, pb) = (&ba.points[i], &bb.points[i]);
                let at = |e: String| format!("branch {id}, point {i}: {e}");
                same_sampled(&pa.lambda, &pb.lambda, o.tolerance, "parameter").map_err(at)?;
                same_sampled(&in_baseline_order(&pa.u), &pb.u, o.tolerance, "state").map_err(at)?;
                if differs(pa.period, pb.period, o.tolerance) {
                    return Err(at(format!("period is {:.12}, baseline has {:.12} (tolerance {:.3e})", pa.period, pb.period, o.tolerance)));
                }
                same_sampled(&pa.u_min, &pb.u_min, o.tolerance, "orbit minimum").map_err(at)?;
                same_sampled(&pa.u_max, &pb.u_max, o.tolerance, "orbit maximum").map_err(at)?;
                // Always include the last point: drift accumulates towards it.
                i = if i + stride >= bb.points.len() && i != bb.points.len() - 1 { bb.points.len() - 1 } else { i + stride };
            }
        }
    }
    Ok(())
}
