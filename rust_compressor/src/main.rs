extern crate chashmap;
extern crate clap;
extern crate itertools;
#[cfg_attr(test, macro_use)]
extern crate polytype;
extern crate programinduction;
extern crate rayon;
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;

mod vs;
use self::vs::induce_version_spaces;

use polytype::Type;
use programinduction::{lambda, ECFrontier, Task};
use rayon::prelude::*;
use std::f64;
use std::io;

#[derive(Copy, Clone, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Strategy {
    VersionSpaces { top_i: usize },
    FragmentGrammars,
}
impl Default for Strategy {
    fn default() -> Strategy {
        Strategy::FragmentGrammars
    }
}

#[derive(Deserialize)]
struct ExternalCompressionInput {
    #[serde(default)]
    strategy: Strategy,
    primitives: Vec<Primitive>,
    #[serde(default)]
    inventions: Vec<Invention>,
    #[serde(default)]
    symmetry_violations: Vec<SymmetryViolation>,
    variable_logprob: f64,
    params: Params,
    frontiers: Vec<Frontier>,
}
#[derive(Serialize)]
struct ExternalCompressionOutput {
    primitives: Vec<Primitive>,
    inventions: Vec<Invention>,
    symmetry_violations: Vec<SymmetryViolation>,
    variable_logprob: f64,
    frontiers: Vec<Frontier>,
}

#[derive(Serialize, Deserialize)]
struct Primitive {
    name: String,
    tp: String,
    #[serde(default)]
    logp: f64,
}

#[derive(Serialize, Deserialize)]
struct Invention {
    expression: String,
    #[serde(default)]
    logp: f64,
}

#[derive(Serialize, Deserialize)]
struct SymmetryViolation {
    f: usize,
    i: usize,
    arg: usize,
}

#[derive(Serialize, Deserialize)]
struct Params {
    pseudocounts: u64,
    topk: usize,
    topk_use_only_likelihood: Option<bool>,
    structure_penalty: f64,
    aic: Option<f64>,
    arity: u32,
}

#[derive(Serialize, Deserialize)]
struct Frontier {
    task_tp: String,
    solutions: Vec<Solution>,
}

#[derive(Serialize, Deserialize)]
struct Solution {
    expression: String,
    logprior: f64,
    loglikelihood: f64,
}

fn noop_oracle(_: &lambda::Language, _: &lambda::Expression) -> f64 {
    f64::NEG_INFINITY
}

struct CompressionInput {
    dsl: lambda::Language,
    params: lambda::CompressionParams,
    tasks: Vec<Task<'static, lambda::Language, lambda::Expression, ()>>,
    frontiers: Vec<ECFrontier<lambda::Language>>,
}
impl From<ExternalCompressionInput> for CompressionInput {
    fn from(eci: ExternalCompressionInput) -> Self {
        let primitives = eci
            .primitives
            .into_par_iter()
            .map(|p| {
                (
                    p.name,
                    Type::parse(&p.tp)
                        .expect("invalid primitive type")
                        .generalize(&[]),
                    p.logp,
                )
            })
            .collect();
        let variable_logprob = eci.variable_logprob;
        let symmetry_violations = eci
            .symmetry_violations
            .into_iter()
            .map(|s| (s.f, s.i, s.arg))
            .collect();
        let mut dsl = lambda::Language {
            primitives,
            invented: vec![],
            variable_logprob,
            symmetry_violations,
        };
        for inv in eci.inventions {
            let expr = dsl.parse(&inv.expression).expect("invalid invention");
            let tp = dsl.infer(&expr).expect("invalid invention type");
            dsl.invented.push((expr, tp, inv.logp))
        }
        let params = lambda::CompressionParams {
            pseudocounts: eci.params.pseudocounts,
            topk: eci.params.topk,
            topk_use_only_likelihood: eci.params.topk_use_only_likelihood.unwrap_or(false),
            structure_penalty: eci.params.structure_penalty,
            aic: eci.params.aic.unwrap_or(f64::INFINITY),
            arity: eci.params.arity,
        };
        let (tasks, frontiers) = eci
            .frontiers
            .into_par_iter()
            .map(|f| {
                let tp = Type::parse(&f.task_tp)
                    .expect("invalid task type")
                    .generalize(&[]);
                let task = Task {
                    oracle: Box::new(noop_oracle),
                    observation: (),
                    tp,
                };
                let sols = f
                    .solutions
                    .into_iter()
                    .map(|s| {
                        let expr = dsl
                            .parse(&s.expression)
                            .expect("invalid expression in frontier");
                        (expr, s.logprior, s.loglikelihood)
                    })
                    .collect();
                (task, ECFrontier(sols))
            })
            .unzip();
        CompressionInput {
            dsl,
            params,
            tasks,
            frontiers,
        }
    }
}
impl From<CompressionInput> for ExternalCompressionOutput {
    fn from(ci: CompressionInput) -> Self {
        let primitives = ci
            .dsl
            .primitives
            .par_iter()
            .map(|&(ref name, ref tp, logp)| Primitive {
                name: name.clone(),
                tp: format!("{}", tp),
                logp,
            })
            .collect();
        let variable_logprob = ci.dsl.variable_logprob;
        let inventions = ci
            .dsl
            .invented
            .par_iter()
            .map(|&(ref expr, _, logp)| Invention {
                expression: ci.dsl.display(expr),
                logp,
            })
            .collect();
        let symmetry_violations = ci
            .dsl
            .symmetry_violations
            .iter()
            .map(|&(f, i, arg)| SymmetryViolation { f, i, arg })
            .collect();
        let frontiers = ci
            .tasks
            .par_iter()
            .zip(&ci.frontiers)
            .map(|(t, f)| {
                let solutions = f
                    .iter()
                    .map(|&(ref expr, logprior, loglikelihood)| {
                        let expression = ci.dsl.display(expr);
                        Solution {
                            expression,
                            logprior,
                            loglikelihood,
                        }
                    })
                    .collect();
                Frontier {
                    task_tp: format!("{}", t.tp),
                    solutions,
                }
            })
            .collect();
        ExternalCompressionOutput {
            primitives,
            inventions,
            variable_logprob,
            symmetry_violations,
            frontiers,
        }
    }
}

fn main() {
    let eci: ExternalCompressionInput = {
        let stdin = io::stdin();
        let handle = stdin.lock();
        serde_json::from_reader(handle).expect("invalid json")
    };
    let strategy = eci.strategy;

    let mut ci = CompressionInput::from(eci);
    let (dsl, frontiers) = match strategy {
        Strategy::FragmentGrammars => ci.dsl.compress(&ci.params, &ci.tasks, ci.frontiers),
        Strategy::VersionSpaces { top_i } => {
            induce_version_spaces(&ci.dsl, &ci.params, &ci.tasks, ci.frontiers, top_i)
        }
    };
    for i in ci.dsl.invented.len()..dsl.invented.len() {
        let &(ref expr, _, _) = &dsl.invented[i];
        eprintln!("invented {}", dsl.display(expr));
    }
    ci.dsl = dsl;
    ci.frontiers = frontiers;
    let eci = ExternalCompressionOutput::from(ci);

    {
        let stdout = io::stdout();
        let handle = stdout.lock();
        serde_json::to_writer(handle, &eci).expect("failed to write result");
    }
}
