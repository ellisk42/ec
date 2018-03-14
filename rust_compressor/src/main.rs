extern crate polytype;
extern crate programinduction;
extern crate rayon;
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;

use std::f64;
use std::io;
use polytype::Type;
use programinduction::{lambda, ECFrontier, Task};
use rayon::prelude::*;

#[derive(Deserialize)]
struct ExternalCompressionInput {
    primitives: Vec<Primitive>,
    inventions: Vec<Invention>,
    variable_logprob: f64,
    params: Params,
    frontiers: Vec<Frontier>,
}
#[derive(Serialize)]
struct ExternalCompressionOutput {
    primitives: Vec<Primitive>,
    inventions: Vec<Invention>,
    variable_logprob: f64,
    frontiers: Vec<Frontier>,
}

#[derive(Serialize, Deserialize)]
struct Primitive {
    name: String,
    tp: String,
    logp: f64,
}

#[derive(Serialize, Deserialize)]
struct Invention {
    expression: String,
    logp: f64,
}

#[derive(Serialize, Deserialize)]
struct Params {
    pseudocounts: u64,
    topk: usize,
    structure_penalty: f64,
    aic: f64,
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
        let primitives = eci.primitives
            .into_par_iter()
            .map(|p| {
                (
                    p.name,
                    Type::parse(&p.tp).expect("invalid primitive type"),
                    p.logp,
                )
            })
            .collect();
        let variable_logprob = eci.variable_logprob;
        let mut dsl = lambda::Language {
            primitives,
            invented: vec![],
            variable_logprob,
        };
        let invented = eci.inventions
            .into_par_iter()
            .map(|inv| {
                let expr = dsl.parse(&inv.expression).expect("invalid invention");
                let tp = dsl.infer(&expr).expect("invalid invention type");
                (expr, tp, inv.logp)
            })
            .collect();
        dsl.invented = invented;
        let params = lambda::CompressionParams {
            pseudocounts: eci.params.pseudocounts,
            topk: eci.params.topk,
            structure_penalty: eci.params.structure_penalty,
            aic: eci.params.aic,
            arity: eci.params.arity,
        };
        let (tasks, frontiers) = eci.frontiers
            .into_par_iter()
            .map(|f| {
                let tp = Type::parse(&f.task_tp).expect("invalid task type");
                let task = Task {
                    oracle: Box::new(noop_oracle),
                    observation: (),
                    tp,
                };
                let sols = f.solutions
                    .into_iter()
                    .map(|s| {
                        let expr = dsl.parse(&s.expression)
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
        let primitives = ci.dsl
            .primitives
            .par_iter()
            .map(|&(ref name, ref tp, logp)| Primitive {
                name: name.clone(),
                tp: format!("{}", tp),
                logp,
            })
            .collect();
        let variable_logprob = ci.dsl.variable_logprob;
        let inventions = ci.dsl
            .invented
            .par_iter()
            .map(|&(ref expr, _, logp)| Invention {
                expression: ci.dsl.display(expr),
                logp,
            })
            .collect();
        let frontiers = ci.tasks
            .par_iter()
            .zip(&ci.frontiers)
            .map(|(t, f)| {
                let solutions = f.iter()
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

    let mut ci = CompressionInput::from(eci);
    let (dsl, frontiers) = ci.dsl.compress(&ci.params, &ci.tasks, ci.frontiers);
    ci.dsl = dsl;
    ci.frontiers = frontiers;
    let eci = ExternalCompressionOutput::from(ci);

    {
        let stdout = io::stdout();
        let handle = stdout.lock();
        serde_json::to_writer_pretty(handle, &eci).expect("failed to write result");
    }
}
