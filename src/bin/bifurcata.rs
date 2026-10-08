//! The console harness (specification §10.1):
//!
//! ```text
//! bifurcata run <model | file.ant> [--param P] [--range lo:hi] [--direction +|-|both]
//!               [--start v1,v2,...] [--settings file] [--max-points n]
//!               [--detect | --no-detect] [--no-spectrum] [--format json|csv]
//!               [-o out.json] [--quiet]
//! bifurcata compare <out.json> <baseline.json> [--tol 1e-6] [--locations-only]
//! bifurcata models
//! ```
//!
//! Exit codes: 0 completed; 1 stopped at a parameter or state bound (normal);
//! 2 continuation failed; 3 no starting point; 4 model load error; 5 compare
//! mismatch.

use std::process::ExitCode;

use bifurcata::continuation::ContinuationOptions;
use bifurcata::models::{BUILTIN_MODELS, builtin_start_point, create_builtin, find_builtin};
use bifurcata::problem::BifurcationProblem;
use bifurcata::run::*;
use bifurcata::serialise::*;

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  bifurcata run <model> [--param P] [--range lo:hi] [--direction +|-|both] [--start v1,v2,...]\n                \
         [--settings file] [--max-points n] [--detect|--no-detect] [--no-spectrum] [--format json|csv] [-o file] [--quiet]\n  \
         bifurcata compare <out.json> <baseline.json> [--tol 1e-6] [--locations-only]\n  bifurcata models"
    );
    ExitCode::from(EXIT_USAGE as u8)
}

fn exit(code: i32) -> ExitCode {
    ExitCode::from(code as u8)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run_command(&args[1..]),
        Some("compare") => compare_command(&args[1..]),
        Some("models") => {
            for m in &BUILTIN_MODELS {
                println!("{:<14} {}\n{:<14} parameter {}, range {}:{}; {}", m.name, m.description, "", m.default_parameter, m.default_range.0, m.default_range.1, m.known_result);
            }
            exit(EXIT_OK)
        }
        _ => usage(),
    }
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))
}

fn compare_command(args: &[String]) -> ExitCode {
    let mut files = Vec::new();
    let mut options = CompareOptions::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--tol" => {
                i += 1;
                match args.get(i).and_then(|v| v.parse().ok()) {
                    Some(t) => options.tolerance = t,
                    None => return usage(),
                }
            }
            "--locations-only" => options.compare_normal_forms = false,
            f => files.push(f.to_owned()),
        }
        i += 1;
    }
    let [actual, baseline] = files.as_slice() else { return usage() };
    let load = |p: &str| read(p).and_then(|t| load_run(&t).map_err(|e| format!("could not parse {p}: {e}")));
    let (a, b) = match (load(actual), load(baseline)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("{e}");
            return exit(EXIT_MODEL_LOAD);
        }
    };
    match compare_runs(&a, &b, &options) {
        Ok(()) => {
            println!("match: {} branches, {} bifurcations, tolerance {}", b.branches.len(), b.total_bifurcations(), options.tolerance);
            exit(EXIT_OK)
        }
        Err(why) => {
            eprintln!("MISMATCH: {why}");
            exit(EXIT_COMPARE_MISMATCH)
        }
    }
}

/// Finds the starting equilibrium at given parameter values.
type StartPoint = Box<dyn Fn(&[f64]) -> Result<Vec<f64>, String>>;

/// A model ready to run: a built-in by name, or (feature `antimony`) a `.ant` file.
struct ResolvedModel {
    problem: Box<dyn BifurcationProblem>,
    lambda0: Vec<f64>,
    name: String,
    source: String,
    default_parameter: Option<String>,
    default_range: Option<(f64, f64)>,
    /// The starting equilibrium at given parameter values.
    start: StartPoint,
}

fn resolve_model(spec: &str) -> Result<ResolvedModel, String> {
    let is_file = spec.ends_with(".ant") || spec.ends_with(".txt");
    if !is_file {
        let (Some(info), Some((problem, lambda0))) = (find_builtin(spec), create_builtin(spec)) else {
            return Err(format!("unknown model \"{spec}\"; try `bifurcata models`"));
        };
        let name = info.name;
        return Ok(ResolvedModel {
            problem,
            lambda0,
            name: name.to_owned(),
            source: format!("builtin:{name}"),
            default_parameter: Some(info.default_parameter.to_owned()),
            default_range: Some(info.default_range),
            start: Box::new(move |lambda| builtin_start_point(name, lambda).ok_or_else(|| format!("no analytic starting point for \"{name}\""))),
        });
    }
    antimony_model(spec)
}

#[cfg(feature = "antimony")]
fn antimony_model(path: &str) -> Result<ResolvedModel, String> {
    use bifurcata::antimony::AntimonyProblem;
    let text = read(path)?;
    let problem = AntimonyProblem::parse(&text).map_err(|e| format!("{path}: {e}"))?;
    let lambda0 = problem.parameter_values();
    let solver = AntimonyProblem::parse(&text)?;
    let name = std::path::Path::new(path).file_stem().map_or_else(|| path.to_owned(), |s| s.to_string_lossy().into_owned());
    Ok(ResolvedModel {
        problem: Box::new(problem),
        lambda0,
        name,
        source: path.to_owned(),
        default_parameter: None,
        default_range: None,
        start: Box::new(move |lambda| solver.find_steady_state(lambda)),
    })
}

#[cfg(not(feature = "antimony"))]
fn antimony_model(path: &str) -> Result<ResolvedModel, String> {
    Err(format!("{path}: reading Antimony needs bifurcata built with the `antimony` feature"))
}

struct RunArgs {
    model: String,
    param: Option<String>,
    range: Option<(f64, f64)>,
    direction: i32,
    start: Option<Vec<f64>>,
    settings: Option<String>,
    max_points: Option<usize>,
    detect: Option<bool>,
    spectrum: bool,
    csv: bool,
    out: Option<String>,
    quiet: bool,
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, String> {
    let mut a = RunArgs {
        model: String::new(),
        param: None,
        range: None,
        direction: 0,
        start: None,
        settings: None,
        max_points: None,
        detect: None,
        spectrum: true,
        csv: false,
        out: None,
        quiet: false,
    };
    let mut i = 0;
    let value = |i: &mut usize, what: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("{what} needs a value"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--param" => a.param = Some(value(&mut i, "--param")?),
            "--range" => {
                let v = value(&mut i, "--range")?;
                let (lo, hi) = v.split_once(':').ok_or("--range expects lo:hi")?;
                a.range = Some((lo.trim().parse().map_err(|_| "--range: bad number")?, hi.trim().parse().map_err(|_| "--range: bad number")?));
            }
            "--direction" => {
                a.direction = match value(&mut i, "--direction")?.as_str() {
                    "+" | "+1" | "1" | "up" => 1,
                    "-" | "-1" | "down" => -1,
                    "both" | "0" => 0,
                    d => return Err(format!("--direction: unknown direction \"{d}\"")),
                }
            }
            "--start" => {
                let v = value(&mut i, "--start")?;
                a.start = Some(v.split(',').map(|s| s.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|_| "--start expects comma-separated numbers")?);
            }
            "--settings" => a.settings = Some(value(&mut i, "--settings")?),
            "--max-points" => a.max_points = Some(value(&mut i, "--max-points")?.parse().map_err(|_| "--max-points: bad number")?),
            "--detect" => a.detect = Some(true),
            "--no-detect" => a.detect = Some(false),
            "--no-spectrum" => a.spectrum = false,
            "--format" => a.csv = value(&mut i, "--format")?.eq_ignore_ascii_case("csv"),
            "-o" => a.out = Some(value(&mut i, "-o")?),
            "--quiet" => a.quiet = true,
            s if s.starts_with('-') => return Err(format!("unknown option {s}")),
            s if a.model.is_empty() => a.model = s.to_owned(),
            s => return Err(format!("unexpected argument {s}")),
        }
        i += 1;
    }
    if a.model.is_empty() {
        return Err("no model given".to_owned());
    }
    Ok(a)
}

fn run_command(args: &[String]) -> ExitCode {
    let a = match parse_run_args(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return usage();
        }
    };
    let model = match resolve_model(&a.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return exit(EXIT_MODEL_LOAD);
        }
    };
    let problem = model.problem.as_ref();
    let lambda0 = model.lambda0.clone();
    let Some(param) = a.param.clone().or(model.default_parameter.clone()) else {
        eprintln!("--param is required for a model file. Available parameters:");
        for (k, value) in lambda0.iter().enumerate() {
            eprintln!("    {:<16} = {value}", problem.parameter_name(k));
        }
        return exit(EXIT_MODEL_LOAD);
    };
    // Parameter names are case-sensitive: "k" and "K" are different parameters.
    let Some(active) = (0..problem.parameter_count()).find(|&k| problem.parameter_name(k) == param) else {
        eprintln!("model \"{}\" has no parameter \"{param}\" (names are case-sensitive)", model.name);
        return exit(EXIT_MODEL_LOAD);
    };
    let Some((lo, hi)) = a.range.or(model.default_range) else {
        eprintln!("--range is required for a model file");
        return exit(EXIT_MODEL_LOAD);
    };

    let u0 = match &a.start {
        Some(u) if u.len() == problem.state_dim() => u.clone(),
        Some(_) => {
            eprintln!("--start expects {} comma-separated values for this model", problem.state_dim());
            return exit(EXIT_NO_START);
        }
        None => match (model.start)(&lambda0) {
            Ok(u) => u,
            Err(e) => {
                eprintln!("starting point: {e}; supply one with --start");
                return exit(EXIT_NO_START);
            }
        },
    };

    // Precedence, weakest first: defaults, a settings file, then flags.
    let mut options = ContinuationOptions::default();
    if let Some(path) = &a.settings {
        match read(path).and_then(|t| apply_options_json(&t, &mut options)) {
            Ok(unknown) => unknown.iter().for_each(|k| eprintln!("warning: settings key \"{k}\" is not recognised")),
            Err(e) => {
                eprintln!("could not read settings: {e}");
                return exit(EXIT_MODEL_LOAD);
            }
        }
    }
    options.parameter_min = lo;
    options.parameter_max = hi;
    if let Some(d) = a.detect {
        options.detect_bifurcations = d;
    }
    if let Some(n) = a.max_points {
        options.max_points = n;
    }

    let run = EquilibriumRun {
        problem,
        model_name: model.name.clone(),
        model_source: model.source.clone(),
        lambda0,
        active,
        u0,
        direction: a.direction,
        options: options.clone(),
    };
    let result = match run.run() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return exit(EXIT_NO_START);
        }
    };

    let output = if a.csv {
        run_to_csv(&result.branches, &result.meta)
    } else {
        run_to_json_string(&result.branches, &result.meta, &options, &SerialiseOptions { include_spectrum: a.spectrum, ..Default::default() })
    };
    match &a.out {
        Some(path) => {
            if let Err(e) = std::fs::write(path, output) {
                eprintln!("cannot write {path}: {e}");
                return exit(EXIT_MODEL_LOAD);
            }
        }
        None => println!("{output}"),
    }

    // A human-readable summary on stderr, so it never contaminates stdout.
    if !a.quiet {
        let total: usize = result.branches.iter().map(|b| b.points.len()).sum();
        eprintln!("{}: {total} points, {param} in [{lo}, {hi}]", model.name);
        for b in &result.branches {
            eprintln!("  branch {}: {} points, {}", b.id, b.points.len(), b.termination.as_str());
            for i in &b.bifurcations {
                eprintln!("  {:<4} {param} = {:.10}   {}", i.kind.abbreviation(), i.point.lambda[active], i.detail);
            }
        }
        for d in &result.declined {
            eprintln!("  declined {} at {param} = {:.10}: {}", d.kind.abbreviation(), d.parameter, d.detail);
        }
        eprintln!("  {}", result.meta.message);
    }
    exit(result.meta.exit_code)
}
