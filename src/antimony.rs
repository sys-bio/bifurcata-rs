//! Antimony models, through websim's model crate (feature `antimony`).
//!
//! The state is the **reduced** state, the independent species only (§3.2): a
//! network with conserved moieties has a structurally singular full Jacobian.
//! The residual is websim's `N_R v(L u + T)` and the Jacobian its `N_R ε L`,
//! with elasticities by finite differences — never symbolic derivatives, which
//! blow up on enzyme rate laws.
//!
//! The parameters are the model's own (rules excluded), in the order the model
//! declares them, followed by one per conservation law, named `_CSUM0`,
//! `_CSUM1`, … as libRoadRunner names them, so a conserved total can be the
//! continuation parameter. Their values are the totals of the initial state.

use websim_model::model::Model;
use websim_model::steady::find_steady_state;

use crate::matrix::Matrix;
use crate::problem::BifurcationProblem;

pub struct AntimonyProblem {
    model: Model,
    /// The model's variable values (`Model::parameter_values`), rule slots included.
    base: Vec<f64>,
    /// For each model parameter exposed here, its slot in `base`.
    slots: Vec<usize>,
    names: Vec<String>,
    totals: Vec<f64>,
    state_names: Vec<String>,
}

impl AntimonyProblem {
    pub fn parse(text: &str) -> Result<Self, String> {
        Ok(Self::new(Model::parse(text).map_err(|e| e.to_string())?))
    }

    pub fn new(model: Model) -> Self {
        let species = model.species.len();
        // symbols() lists the species, then the parameters (rules excluded).
        let mut names: Vec<String> = model.symbols().skip(species).map(|s| s.name.clone()).collect();
        let slots = names.iter().map(|n| model.parameter_index(n).expect("a listed parameter has a slot")).collect();
        let conservation = model.conservation();
        let totals = conservation.totals(&model.initial_state());
        names.extend((0..totals.len()).map(|k| format!("_CSUM{k}")));
        let all = model.species_names();
        let state_names = conservation.independent.iter().map(|&i| all[i].clone()).collect();
        Self { base: model.parameter_values(), slots, names, totals, state_names, model }
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Every parameter's current value, conserved totals last.
    pub fn parameter_values(&self) -> Vec<f64> {
        self.slots.iter().map(|&s| self.base[s]).chain(self.totals.iter().copied()).collect()
    }

    /// The index of a parameter by name (case-sensitive: `k` and `K` differ).
    pub fn parameter_index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    pub fn is_conserved_total(&self, k: usize) -> bool {
        k >= self.slots.len()
    }

    /// The model's variable vector and the conserved totals for `lambda`.
    fn split<'a>(&self, lambda: &'a [f64]) -> (Vec<f64>, &'a [f64]) {
        let mut params = self.base.clone();
        for (&slot, v) in self.slots.iter().zip(lambda) {
            params[slot] = *v;
        }
        (params, &lambda[self.slots.len()..])
    }

    /// A steady state at `lambda` by websim's solver (Newton, then integration
    /// to an attractor and polishing), from the model's initial values with the
    /// totals `lambda` sets. Returns the independent species.
    pub fn find_steady_state(&self, lambda: &[f64]) -> Result<Vec<f64>, String> {
        let (params, totals) = self.split(lambda);
        let conservation = self.model.conservation();
        let x0 = conservation.full_state(&conservation.reduce(&self.model.initial_state()), totals);
        Ok(find_steady_state(&self.model, &x0, &params)?.reduced)
    }
}

impl BifurcationProblem for AntimonyProblem {
    fn state_dim(&self) -> usize {
        self.state_names.len()
    }

    fn parameter_count(&self) -> usize {
        self.names.len()
    }

    fn residual(&self, u: &[f64], lambda: &[f64], r: &mut [f64]) {
        let (params, totals) = self.split(lambda);
        r.copy_from_slice(&self.model.reduced_rates(0.0, u, totals, &params));
    }

    fn jacobian(&self, u: &[f64], lambda: &[f64], j: &mut Matrix) {
        let (params, totals) = self.split(lambda);
        let jr = self.model.reduced_jacobian(0.0, u, totals, &params);
        *j = Matrix::from_fn(jr.rows(), jr.cols(), |a, b| jr[(a, b)]);
    }

    fn state_name(&self, i: usize) -> String {
        self.state_names[i].clone()
    }

    fn parameter_name(&self, k: usize) -> String {
        self.names[k].clone()
    }
}
