//! The `[bifurcation]` block a model can carry in a comment (specification
//! §13.5), saying how it is meant to be continued:
//!
//! ```text
//! /*
//! [bifurcation]
//! parameter: m
//! min: 0
//! max: 2
//! ds: 0.001
//! dsMax: 0.005
//! maxSteps: 10000
//! plot: CycBT
//! */
//! ```
//!
//! Without it a model is half a specification: the equations say what the
//! system is, nothing says which parameter matters or over what range — and
//! the step sizes that make a stiff model behave belong with the model, not in
//! the head of whoever ran it last.
//!
//! A blank value means "not specified", not zero. Unknown keys are collected:
//! a typo that silently does nothing looks exactly like an option without
//! effect. Precedence, weakest first: built-in defaults, this block, a
//! settings file, command-line flags — the block is a default, not a mandate.
//! Ported from the Delphi `Bifurcata.RunSpec`.

use crate::continuation::ContinuationOptions;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunSpec {
    /// Whether a `[bifurcation]` block was present at all.
    pub found: bool,
    pub parameter: Option<String>,
    /// Only when both `min` and `max` are given, with min < max.
    pub range: Option<(f64, f64)>,
    /// A starting value for the parameter.
    pub start: Option<f64>,
    /// The variable to plot (`plot`, `variable` or `y`).
    pub plot: Option<String>,
    /// The initial step (`ds`).
    pub ds: Option<f64>,
    /// The maximum step (`dsMax`).
    pub ds_max: Option<f64>,
    /// The point budget (`maxSteps` or `maxPoints`).
    pub max_steps: Option<usize>,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
    pub unknown_keys: Vec<String>,
}

impl RunSpec {
    /// Read the block from a model's text; an empty spec if there is none.
    pub fn parse(source: &str) -> RunSpec {
        let mut spec = RunSpec::default();
        let (mut min, mut max) = (None, None);
        let mut in_block = false;
        for line in source.lines() {
            let raw = line.trim();
            // The header can sit inside a /* */ comment, on its own line.
            if raw.eq_ignore_ascii_case("[bifurcation]") {
                in_block = true;
                spec.found = true;
                continue;
            }
            if !in_block {
                continue;
            }
            // The block ends at the comment terminator, a new [section], or
            // the first line that is not `key: value`.
            if raw.is_empty() {
                continue;
            }
            if raw == "*/" || raw.starts_with('[') {
                in_block = false;
                continue;
            }
            let Some((key, value)) = raw.split_once(':') else {
                in_block = false;
                continue;
            };
            let (key, value) = (key.trim(), value.trim());
            // `x := ...` (a rule) or `R1: -> X; k` (a reaction) is model
            // text, not a key: the block is over.
            if value.starts_with('=') || value.contains("->") || value.contains(';') {
                in_block = false;
                continue;
            }
            if value.is_empty() {
                continue;
            }
            let number = value.parse::<f64>().ok();
            match key.to_ascii_lowercase().as_str() {
                "parameter" => spec.parameter = Some(value.to_owned()),
                "min" => min = number,
                "max" => max = number,
                "start" => spec.start = number,
                "plot" | "variable" | "y" => spec.plot = Some(value.to_owned()),
                "ds" => spec.ds = number,
                "dsmax" => spec.ds_max = number,
                "maxsteps" | "maxpoints" => spec.max_steps = value.parse().ok(),
                "ymin" => spec.y_min = number,
                "ymax" => spec.y_max = number,
                _ => spec.unknown_keys.push(key.to_owned()),
            }
        }
        if let (Some(lo), Some(hi)) = (min, max)
            && lo < hi
        {
            spec.range = Some((lo, hi));
        }
        spec
    }

    /// Apply what the block specifies onto `options`, leaving the rest.
    pub fn apply_to(&self, options: &mut ContinuationOptions) {
        if let Some((lo, hi)) = self.range {
            options.parameter_min = lo;
            options.parameter_max = hi;
        }
        if let Some(ds) = self.ds {
            options.initial_step = ds;
        }
        if let Some(ds_max) = self.ds_max {
            options.max_step = ds_max;
        }
        if let Some(n) = self.max_steps {
            options.max_points = n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_block_inside_a_comment() {
        let spec = RunSpec::parse(
            "/*\n[bifurcation]\nparameter: m\nmin: 0\nmax: 2\nstart:\nds: 0.001\ndsMax: 0.005\nmaxSteps: 10000\nplot: CycBT\nymin:\n*/\n\
             sig:=CycBT+CKIT\n-> X; k\n",
        );
        assert!(spec.found);
        assert_eq!(spec.parameter.as_deref(), Some("m"));
        assert_eq!(spec.range, Some((0.0, 2.0)));
        assert_eq!(spec.start, None, "a blank value is not specified, not zero");
        assert_eq!((spec.ds, spec.ds_max, spec.max_steps), (Some(0.001), Some(0.005), Some(10000)));
        assert_eq!(spec.plot.as_deref(), Some("CycBT"));
        assert!(spec.unknown_keys.is_empty(), "nothing after */ is read as a key: {:?}", spec.unknown_keys);

        let mut options = ContinuationOptions::default();
        spec.apply_to(&mut options);
        assert_eq!((options.parameter_min, options.parameter_max, options.initial_step, options.max_step, options.max_points), (0.0, 2.0, 0.001, 0.005, 10000));
        assert_eq!(options.tol_residual, ContinuationOptions::default().tol_residual, "the rest is left alone");
    }

    #[test]
    fn absent_partial_and_mistyped() {
        let none = RunSpec::parse("-> X; k\nk = 1\n");
        assert!(!none.found);
        assert_eq!(none, RunSpec::default());

        // A range needs both ends; a typo is reported, not ignored.
        let spec = RunSpec::parse("// [bifurcation]\n[bifurcation]\nparameter: k\nmin: 1\nmxa: 3\n\n-> X; k\n");
        assert_eq!(spec.range, None, "min alone is not a range");
        assert_eq!(spec.unknown_keys, vec!["mxa"]);

        // An assignment rule or a named reaction straight after the block ends it.
        let spec = RunSpec::parse("[bifurcation]\nparameter: k\ns := X + Y\n");
        assert!(spec.unknown_keys.is_empty());
        assert_eq!(spec.parameter.as_deref(), Some("k"));
        let spec = RunSpec::parse("[bifurcation]\nparameter: k\nR1: -> X; k\nmax: 3\n");
        assert!(spec.unknown_keys.is_empty(), "{:?}", spec.unknown_keys);
        assert_eq!(spec.range, None, "nothing after the reaction is read");
    }
}
