//! The built-in models of the console harness (the Delphi `Bifurcata.Models`):
//! the problems with closed-form answers that every baseline run starts from.

use crate::problem::BifurcationProblem;
use crate::test_problems::{Brusselator, FoldProblem, Selkov, TranscriticalProblem};

pub struct BuiltinModel {
    pub name: &'static str,
    pub description: &'static str,
    pub default_parameter: &'static str,
    pub default_range: (f64, f64),
    pub known_result: &'static str,
}

pub const BUILTIN_MODELS: [BuiltinModel; 4] = [
    BuiltinModel {
        name: "brusselator",
        description: "Brusselator: A - (B+1)X + X^2 Y, B X - X^2 Y",
        default_parameter: "B",
        default_range: (1.0, 4.0),
        known_result: "supercritical Hopf at B = 1 + A^2 (= 2 for A = 1), omega = A",
    },
    BuiltinModel {
        name: "selkov",
        description: "Selkov glycolysis: v0 - S P^2, S P^2 - k P",
        default_parameter: "v0",
        default_range: (0.02, 1.2),
        known_result: "Hopf at v0 = k^(3/2) (= 0.4647580015 for k = 0.6), omega = k",
    },
    BuiltinModel {
        name: "saddlenode",
        description: "Saddle-node: lambda - u^2",
        default_parameter: "lambda",
        default_range: (-0.5, 4.0),
        known_result: "fold at lambda = 0, u = 0",
    },
    BuiltinModel {
        name: "transcritical",
        description: "Transcritical: lambda u - u^2",
        default_parameter: "lambda",
        default_range: (-1.5, 1.5),
        known_result: "branch point at lambda = 0, u = 0",
    },
];

/// The catalogue entry, ignoring case.
pub fn find_builtin(name: &str) -> Option<&'static BuiltinModel> {
    BUILTIN_MODELS.iter().find(|m| m.name.eq_ignore_ascii_case(name))
}

/// The model and its default parameter values.
pub fn create_builtin(name: &str) -> Option<(Box<dyn BifurcationProblem>, Vec<f64>)> {
    Some(match find_builtin(name)?.name {
        "brusselator" => (Box::new(Brusselator) as Box<dyn BifurcationProblem>, vec![1.0, 1.2]),
        "selkov" => (Box::new(Selkov), vec![0.05, 0.6]),
        "saddlenode" => (Box::new(FoldProblem), vec![2.0]),
        "transcritical" => (Box::new(TranscriticalProblem), vec![-1.0]),
        _ => return None,
    })
}

/// The analytic equilibrium at `lambda`, where there is one.
pub fn builtin_start_point(name: &str, lambda: &[f64]) -> Option<Vec<f64>> {
    match find_builtin(name)?.name {
        "brusselator" => Some(Brusselator::equilibrium(lambda[0], lambda[1]).to_vec()),
        "selkov" => Some(Selkov::equilibrium(lambda[0], lambda[1]).to_vec()),
        "saddlenode" => (lambda[0] >= 0.0).then(|| vec![lambda[0].sqrt()]),
        "transcritical" => Some(vec![0.0]),
        _ => None,
    }
}
