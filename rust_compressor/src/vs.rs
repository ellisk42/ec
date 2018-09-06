use chashmap::CHashMap;
use itertools::Itertools;
use programinduction::{
    lambda::{self, Expression},
    ECFrontier, Task,
};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::f64;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

const EPSILON: f64 = 0.001;

pub fn induce_version_spaces<O: Sync>(
    dsl: &lambda::Language,
    params: &lambda::CompressionParams,
    tasks: &[Task<lambda::Language, lambda::Expression, O>],
    original_frontiers: Vec<ECFrontier<lambda::Language>>,
    top_i: usize,
) -> (lambda::Language, Vec<ECFrontier<lambda::Language>>) {
    induce_version_spaces_vt(
        dsl,
        params,
        tasks,
        original_frontiers,
        top_i,
        &VersionTable::new(),
    )
}

pub fn induce_version_spaces_vt<O: Sync>(
    dsl: &lambda::Language,
    params: &lambda::CompressionParams,
    tasks: &[Task<lambda::Language, lambda::Expression, O>],
    original_frontiers: Vec<ECFrontier<lambda::Language>>,
    top_i: usize,
    vt: &VersionTable,
) -> (lambda::Language, Vec<ECFrontier<lambda::Language>>) {
    lambda::induce(
        dsl,
        params,
        tasks,
        original_frontiers,
        vt,
        |vt, _dsl, frontiers, params, out| {
            eprintln!("VERSION_SPACES: generating proposals..");
            let versions: Vec<Vec<_>> = frontiers
                .into_par_iter()
                .map(|fs| {
                    fs.1
                        .iter()
                        .map(|f| {
                            let vs = vt.incorporate(f.0.clone());
                            vt.super_version_space(&vs, params.arity)
                        })
                        .collect()
                })
                .collect();
            eprintln!(
                "VERSION_SPACES: collecting {} best proposals from {}..",
                top_i,
                vt.len()
            );
            let mut best = vt.best_inventions(&versions, top_i);
            out.append(&mut best)
        },
        |vt, candidate, dsl, frontiers, params| {
            let expr = candidate.extract().pop().unwrap();
            if let Err(e) = dsl.invent(expr.clone(), 0.0) {
                eprintln!(
                    "VERSION_SPACES: could not invent VS {}: {}",
                    candidate.0.display(dsl),
                    e
                );
                return None;
            }
            let mut frontiers = frontiers.to_vec();
            rewrite_frontiers(vt, candidate.clone(), expr, dsl, &mut frontiers, params);
            let joint_mdl = dsl.inside_outside(&frontiers, params.pseudocounts);
            eprintln!(
                "VERSION_SPACES: proposed VS {} with joint_mdl={}",
                candidate.0.display(dsl),
                joint_mdl
            );
            Some(joint_mdl)
        },
        |expr| close_invention(expr).0,
        rewrite_frontiers,
    )
}

#[cfg_attr(feature = "cargo-clippy", allow(needless_pass_by_value, trivially_copy_pass_by_ref))]
fn rewrite_frontiers(
    vt: &&VersionTable,
    candidate: VersionSpace,
    nonclosed_invention: Expression,
    _dsl: &lambda::Language,
    frontiers: &mut Vec<lambda::RescoredFrontier>,
    params: &lambda::CompressionParams,
) {
    let rewrite_mapping: Vec<Expression> = frontiers
        .iter()
        .flat_map(|fs| &fs.1)
        .map(|f| &f.0)
        .unique()
        .cloned()
        .collect();
    let spaces = rewrite_mapping
        .par_iter()
        .map(|p| {
            let vs = vt.incorporate(p.clone());
            vt.super_version_space(&vs, params.arity)
        })
        .collect();
    let mut rewrite_mapping: HashMap<_, _> = rewrite_mapping
        .into_par_iter()
        .zip(vt.rewrite_with_invention(&candidate, spaces))
        .collect();
    let (inv, mapping) = close_invention(nonclosed_invention.clone());
    let mapping: HashMap<_, _> = mapping.into_iter().map(|(a, b)| (b, a)).collect();
    let mut applied_invention = inv;
    for j in (0..mapping.len()).rev() {
        applied_invention = Expression::Application(
            Box::new(applied_invention),
            Box::new(Expression::Index(mapping[&j])),
        )
    }
    for mut fs in frontiers {
        for mut f in &mut fs.1 {
            let e = rewrite_mapping.get_mut(&f.0).unwrap();
            rewrite(&nonclosed_invention, &applied_invention, e);
            // e.etalong(dsl);
        }
    }
}

/// Do not create these yourself! Use the VersionTable methods (that start with `vs_`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum VS {
    Union(Vec<Arc<VS>>),
    Terminal(lambda::Expression),
    Application(Arc<VS>, Arc<VS>),
    Abstraction(Arc<VS>),
    Index(usize),
    Universe,
    Void,
}
impl VS {
    fn display(&self, dsl: &lambda::Language) -> String {
        self.show(dsl, false)
    }
    fn show(&self, dsl: &lambda::Language, is_function: bool) -> String {
        match self {
            VS::Union(vs) => {
                let mut s = String::from("Union([");
                for x in vs {
                    s.push_str(&x.show(dsl, false));
                    s.push(' ');
                }
                s.pop();
                s.push_str("])");
                s
            }
            VS::Terminal(e) => dsl.display(&e),
            VS::Application(f, x) => if is_function {
                format!("{} {}", f.show(dsl, true), x.show(dsl, false))
            } else {
                format!("({} {})", f.show(dsl, true), x.show(dsl, false))
            },
            VS::Abstraction(b) => format!("(λ {})", b.show(dsl, false)),
            VS::Index(i) => format!("${}", i),
            VS::Universe => String::from("universe"),
            VS::Void => String::from("void"),
        }
    }
}

/// `VersionSpace` is a wrapper around `VS` that represents a version space known to some
/// `VersionTable`. Every child of a `VersionSpace` is also a `VersionSpace` in that it is known to
/// the `VersionTable`. Therefore a `VS` should _never_ be explicitly constructed (and subsequently
/// turned into a `VersionSpace`. Just use the public methods and everything will work as intended.
#[derive(Clone)]
pub struct VersionSpace(Arc<VS>);
impl VersionSpace {
    pub fn reachable(&self, out: &mut HashSet<VersionSpace>) {
        if out.contains(self) {
            return;
        }
        match *self.0 {
            VS::Union(ref inner) => for vs in inner {
                VersionSpace::from(vs).reachable(out)
            },
            VS::Application(ref f, ref x) => {
                VersionSpace::from(f).reachable(out);
                VersionSpace::from(x).reachable(out);
            }
            VS::Abstraction(ref body) => VersionSpace::from(body).reachable(out),
            _ => (),
        };
        out.insert(self.clone());
    }
    pub fn extract(&self) -> Vec<lambda::Expression> {
        VersionSpace::extract_inner(&self.0)
    }
    fn extract_inner(vs: &Arc<VS>) -> Vec<lambda::Expression> {
        match **vs {
            VS::Union(ref inner) => inner
                .into_iter()
                .flat_map(VersionSpace::extract_inner)
                .collect(),
            VS::Application(ref f, ref x) => VersionSpace::extract_inner(f)
                .into_iter()
                .cartesian_product(VersionSpace::extract_inner(x))
                .map(|(f, x)| Expression::Application(Box::new(f), Box::new(x)))
                .collect(),
            VS::Index(i) => vec![Expression::Index(i)],
            VS::Void => Vec::new(),
            VS::Terminal(ref e) => vec![e.clone()],
            VS::Abstraction(ref body) => VersionSpace::extract_inner(body)
                .into_iter()
                .map(|b| Expression::Abstraction(Box::new(b)))
                .collect(),
            VS::Universe => {
                eprintln!("extracted universe");
                vec![Expression::Primitive(-1i32 as usize)]
            }
        }
    }
}
impl From<Arc<VS>> for VersionSpace {
    fn from(vs: Arc<VS>) -> VersionSpace {
        VersionSpace(vs)
    }
}
impl<'a> From<&'a Arc<VS>> for VersionSpace {
    fn from(vs: &'a Arc<VS>) -> VersionSpace {
        VersionSpace(Arc::clone(vs))
    }
}
impl PartialEq for VersionSpace {
    fn eq(&self, other: &VersionSpace) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for VersionSpace {}
impl PartialOrd for VersionSpace {
    fn partial_cmp(&self, other: &VersionSpace) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for VersionSpace {
    fn cmp(&self, other: &VersionSpace) -> Ordering {
        let a = Arc::into_raw(Arc::clone(&self.0));
        let b = Arc::into_raw(Arc::clone(&other.0));
        let o = a.cmp(&b);
        unsafe {
            // to prevent leak, this will drop the Arc count
            Arc::from_raw(a);
            Arc::from_raw(b);
        }
        o
    }
}
impl Hash for VersionSpace {
    fn hash<H: Hasher>(&self, h: &mut H) {
        let ptr = Arc::into_raw(Arc::clone(&self.0));
        ptr.hash(h);
        unsafe {
            // to prevent leak, this will drop the Arc count
            Arc::from_raw(ptr);
        }
    }
}

#[derive(Clone)]
struct RWCost {
    f: Option<(lambda::Expression, f64)>,
    a: Option<(lambda::Expression, f64)>,
}
impl RWCost {
    #[inline(always)]
    fn fc(&self) -> f64 {
        if let Some((_, c)) = self.f {
            c
        } else {
            f64::INFINITY
        }
    }
    #[inline(always)]
    fn ac(&self) -> f64 {
        if let Some((_, c)) = self.a {
            c
        } else {
            f64::INFINITY
        }
    }
}

/// metadata for a version space
struct VersionMeta {
    vs: Weak<VS>,
    inverse: Option<VersionSpace>,
    substitutions: HashMap<usize, Arc<HashMap<VersionSpace, VersionSpace>>>,
    inhabitants: Option<(f64, Arc<HashSet<VersionSpace>>)>,
    function_inhabitants: Option<(f64, Arc<HashSet<VersionSpace>>)>,
    super_space: Option<VersionSpace>,
}
impl VersionMeta {
    pub fn new(parent: &Arc<VS>) -> VersionMeta {
        VersionMeta {
            vs: Arc::downgrade(parent),
            inverse: None,
            substitutions: HashMap::new(),
            inhabitants: None,
            function_inhabitants: None,
            super_space: None,
        }
    }
}

pub struct VersionTable {
    void: VersionSpace,
    universe: VersionSpace,
    all: CHashMap<Arc<VS>, VersionMeta>,
    overlaps: CHashMap<(VersionSpace, VersionSpace), bool>,
}
impl VersionTable {
    pub fn new() -> VersionTable {
        let void = Arc::new(VS::Void);
        let universe = Arc::new(VS::Universe);
        VersionTable {
            void: VersionSpace::from(&void),
            universe: VersionSpace::from(&universe),
            all: vec![
                (Arc::clone(&void), VersionMeta::new(&void)),
                (Arc::clone(&universe), VersionMeta::new(&universe)),
            ].into_iter()
                .collect(),
            overlaps: CHashMap::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.all.len()
    }
    pub fn vs_apply(&self, f: VersionSpace, x: VersionSpace) -> VersionSpace {
        match (&*f.0, &*x.0) {
            (VS::Void, _) => x,
            (_, VS::Void) => f,
            _ => self.incorporate_space(VS::Application(f.0, x.0)),
        }
    }
    pub fn vs_abstract(&self, body: VersionSpace) -> VersionSpace {
        match *body.0 {
            VS::Void => body,
            _ => self.incorporate_space(VS::Abstraction(body.0)),
        }
    }
    pub fn vs_index(&self, idx: usize) -> VersionSpace {
        self.incorporate_space(VS::Index(idx))
    }
    pub fn vs_terminal(&self, terminal: lambda::Expression) -> VersionSpace {
        self.incorporate_space(VS::Terminal(terminal))
    }
    pub fn incorporate(&self, e: Expression) -> VersionSpace {
        match e {
            lambda::Expression::Index(i) => self.vs_index(i),
            lambda::Expression::Abstraction(body) => {
                let body = self.incorporate(*body);
                self.vs_abstract(body)
            }
            lambda::Expression::Application(f, x) => {
                let f = self.incorporate(*f);
                let x = self.incorporate(*x);
                self.vs_apply(f, x)
            }
            lambda::Expression::Primitive(_) | lambda::Expression::Invented(_) => {
                self.vs_terminal(e)
            }
        }
    }
    pub fn shift(&self, vs: &VersionSpace, offset: i64) -> VersionSpace {
        self.shift_inner(&vs.0, offset, 0)
    }
    fn shift_inner(&self, vs: &Arc<VS>, offset: i64, bound: usize) -> VersionSpace {
        match **vs {
            VS::Union(ref inner) => {
                let inner = inner
                    .into_iter()
                    .map(|vs| self.shift_inner(vs, offset, bound))
                    .collect();
                self.union(inner)
            }
            VS::Index(i) if i < bound => VersionSpace::from(vs),
            VS::Index(i) if i >= (bound as i64 + offset) as usize => {
                self.vs_index((i as i64 - offset) as usize)
            }
            VS::Index(_) => self.void.clone(),
            VS::Application(ref f, ref x) => {
                let f = self.shift_inner(f, offset, bound);
                let x = self.shift_inner(x, offset, bound);
                self.vs_apply(f, x)
            }
            VS::Abstraction(ref b) => {
                let b = self.shift_inner(b, offset, bound + 1);
                self.vs_abstract(b)
            }
            VS::Terminal(_) | VS::Universe | VS::Void => VersionSpace::from(vs),
        }
    }
    pub fn union(&self, vss: Vec<VersionSpace>) -> VersionSpace {
        if vss.contains(&self.universe) {
            self.universe.clone()
        } else {
            let mut merged = vss
                .into_iter()
                .flat_map(|vs| match *vs.0 {
                    // these clones are shallow
                    VS::Union(ref inner) => inner.clone(),
                    VS::Void => vec![],
                    VS::Universe => unreachable!(),
                    _ => vec![Arc::clone(&vs.0)],
                })
                .map(VersionSpace::from)
                .collect::<Vec<_>>();
            match merged.len() {
                0 => self.void.clone(),
                1 => merged.pop().unwrap(),
                _ => {
                    merged.sort_unstable();
                    merged.dedup();
                    let merged = merged.into_iter().map(|vs| vs.0).collect::<Vec<_>>();
                    self.incorporate_space(VS::Union(merged))
                }
            }
        }
    }
    pub fn intersection(&self, a: &VersionSpace, b: &VersionSpace) -> VersionSpace {
        self.intersection_inner(&a.0, &b.0)
    }
    fn intersection_inner(&self, a: &Arc<VS>, b: &Arc<VS>) -> VersionSpace {
        match (&**a, &**b) {
            (VS::Universe, _) => VersionSpace::from(b),
            (_, VS::Universe) => VersionSpace::from(a),
            (VS::Void, _) | (_, VS::Void) => self.void.clone(),
            (VS::Union(ref aa), VS::Union(ref bb)) => {
                let xs = aa
                    .into_iter()
                    .cartesian_product(bb)
                    .map(|(a, b)| self.intersection_inner(a, b))
                    .collect();
                self.union(xs)
            }
            (VS::Union(ref aa), _) => {
                let xs = aa
                    .into_iter()
                    .map(|a| self.intersection_inner(a, b))
                    .collect();
                self.union(xs)
            }
            (_, VS::Union(ref bb)) => {
                let xs = bb
                    .into_iter()
                    .map(|b| self.intersection_inner(a, b))
                    .collect();
                self.union(xs)
            }
            (VS::Abstraction(ref a), VS::Abstraction(ref b)) => {
                let body = self.intersection_inner(a, b);
                self.vs_abstract(body)
            }
            (VS::Application(ref f1, ref x1), VS::Application(ref f2, ref x2)) => {
                let f = self.intersection_inner(f1, f2);
                let x = self.intersection_inner(x1, x2);
                self.vs_apply(f, x)
            }
            (VS::Index(i1), VS::Index(i2)) if i1 == i2 => VersionSpace::from(a),
            (VS::Terminal(ref t1), VS::Terminal(ref t2)) if t1 == t2 => VersionSpace::from(a),
            _ => self.void.clone(),
        }
    }
    pub fn has_intersection(&self, a: &VersionSpace, b: &VersionSpace) -> bool {
        match (&*a.0, &*b.0) {
            (VS::Void, _) | (_, VS::Void) => return false,
            (VS::Universe, _) | (_, VS::Universe) => return true,
            _ => (),
        }
        let k = match a.cmp(&b) {
            Ordering::Equal => return true,
            Ordering::Less => (a.clone(), b.clone()),
            Ordering::Greater => (b.clone(), a.clone()),
        };
        if let Some(x) = self.overlaps.get(&k) {
            return *x;
        }
        let overlap = match (&*a.0, &*b.0) {
            (VS::Union(ref aa), VS::Union(ref bb)) => aa.into_iter().cartesian_product(bb).any(
                |(a, b)| self.has_intersection(&VersionSpace::from(a), &VersionSpace::from(b)),
            ),
            (VS::Union(ref aa), _) => aa
                .into_iter()
                .any(|a| self.has_intersection(&VersionSpace::from(a), b)),
            (_, VS::Union(ref bb)) => bb
                .into_iter()
                .any(|b| self.has_intersection(a, &VersionSpace::from(b))),
            (VS::Abstraction(ref a), VS::Abstraction(ref b)) => {
                self.has_intersection(&VersionSpace::from(a), &VersionSpace::from(b))
            }
            (VS::Application(ref f1, ref x1), VS::Application(ref f2, ref x2)) => {
                self.has_intersection(&VersionSpace::from(f1), &VersionSpace::from(f2))
                    && self.has_intersection(&VersionSpace::from(x1), &VersionSpace::from(x2))
            }
            (VS::Index(i1), VS::Index(i2)) if i1 == i2 => true,
            (VS::Terminal(ref t1), VS::Terminal(ref t2)) if t1 == t2 => true,
            _ => false,
        };
        self.overlaps.insert(k, overlap);
        overlap
    }
    pub fn substitutions(&self, vs: &VersionSpace) -> Arc<HashMap<VersionSpace, VersionSpace>> {
        self.substitutions_inner(vs, 0)
    }
    fn substitutions_inner(
        &self,
        vs: &VersionSpace,
        offset: usize,
    ) -> Arc<HashMap<VersionSpace, VersionSpace>> {
        if self
            .all
            .get(&vs.0)
            .unwrap()
            .substitutions
            .contains_key(&offset)
        {
            return Arc::clone(&self.all.get(&vs.0).unwrap().substitutions[&offset]);
        }
        let mut sub = HashMap::new();
        let shifted = self.shift(&vs, offset as i64);
        if shifted != self.void {
            sub.insert(shifted, self.vs_index(offset));
        }
        match *vs.0 {
            VS::Terminal(_) => {
                sub.insert(self.universe.clone(), vs.clone());
            }
            VS::Index(i) => {
                sub.insert(
                    self.universe.clone(),
                    if i < offset {
                        vs.clone()
                    } else {
                        self.vs_index(1 + i)
                    },
                );
            }
            VS::Abstraction(ref b) => {
                for (k, v) in &*self.substitutions_inner(&VersionSpace::from(b), offset + 1) {
                    let k = k.clone();
                    let v = self.vs_abstract(v.clone());
                    sub.insert(k, v);
                }
            }
            VS::Union(ref u) => {
                let mut new_mapping: HashMap<VersionSpace, Vec<VersionSpace>> = HashMap::new();
                for x in u {
                    for (k, v) in &*self.substitutions_inner(&VersionSpace::from(x), offset) {
                        let k = k.clone();
                        let v = v.clone();
                        new_mapping.entry(k).or_default().push(v)
                    }
                }
                for (k, xs) in new_mapping {
                    sub.insert(k, self.union(xs));
                }
            }
            VS::Application(ref f, ref x) => {
                let mut new_mapping: HashMap<VersionSpace, Vec<VersionSpace>> = HashMap::new();
                let fm = self.substitutions_inner(&VersionSpace::from(f), offset);
                let xm = self.substitutions_inner(&VersionSpace::from(x), offset);
                for (v1, f) in &*fm {
                    for (v2, x) in &*xm {
                        if self.has_intersection(v1, v2) {
                            let v = self.intersection(v1, v2);
                            let a = self.vs_apply(f.clone(), x.clone());
                            new_mapping.entry(v).or_default().push(a);
                        }
                    }
                }
                for (k, xs) in new_mapping {
                    sub.insert(k, self.union(xs));
                }
            }
            _ => (),
        }
        let mut meta = self.all.get_mut(&vs.0).unwrap();
        meta.substitutions.insert(offset, Arc::new(sub));
        Arc::clone(&meta.substitutions[&offset])
    }
    pub fn inversion(&self, vs: &VersionSpace) -> VersionSpace {
        if let Some(ref ri) = self.all.get(&vs.0).unwrap().inverse {
            return ri.clone();
        }
        let ri = match *vs.0 {
            VS::Union(ref inner) => {
                let inversions = inner
                    .into_iter()
                    .map(|x| self.inversion(&VersionSpace::from(x)))
                    .collect();
                self.union(inversions)
            }
            _ => {
                let mut inversions: Vec<_> = self
                    .substitutions(&vs)
                    .iter()
                    .filter_map(|(v, b)| match (&*v.0, &*b.0) {
                        (VS::Universe, _) | (_, VS::Index(0)) => None,
                        _ => {
                            let b = self.vs_abstract(b.clone());
                            Some(self.vs_apply(b, v.clone()))
                        }
                    })
                    .collect();
                match *vs.0 {
                    VS::Application(ref f, ref x) => {
                        let f = VersionSpace::from(f);
                        let x = VersionSpace::from(x);
                        let rif = self.inversion(&f);
                        let rix = self.inversion(&x);
                        inversions.push(self.vs_apply(rif, x));
                        inversions.push(self.vs_apply(f, rix));
                    }
                    VS::Abstraction(ref b) => {
                        let rib = self.inversion(&VersionSpace::from(b));
                        inversions.push(self.vs_abstract(rib));
                    }
                    _ => (),
                }
                self.union(inversions)
            }
        };
        let mut meta = self.all.get_mut(&vs.0).unwrap();
        meta.inverse = Some(ri.clone());
        ri
    }
    fn minimal_inhabitants(&self, vs: &VersionSpace) -> (f64, Arc<HashSet<VersionSpace>>) {
        self.minimal_inhabitants_inner(&vs.0)
    }
    fn minimal_inhabitants_inner(&self, vs: &Arc<VS>) -> (f64, Arc<HashSet<VersionSpace>>) {
        if let Some(ref x) = self.all.get(vs).unwrap().inhabitants {
            return x.clone();
        }
        let (cost, members) = match **vs {
            VS::Union(ref u) => {
                let mut cost = f64::INFINITY;
                let mut members = HashSet::new();
                for child in u {
                    let (subcost, submembers) = self.minimal_inhabitants_inner(child);
                    match subcost.partial_cmp(&cost) {
                        Some(Ordering::Less) => {
                            cost = subcost;
                            members = (*submembers).clone();
                        }
                        Some(Ordering::Equal) => members.extend(submembers.iter().cloned()),
                        _ => (),
                    }
                }
                (cost, members)
            }
            VS::Application(ref f, ref x) => {
                let (fc, fm) = self.minimal_function_inhabitants_inner(f);
                let (xc, xm) = self.minimal_inhabitants_inner(x);
                let cost = fc + xc + EPSILON;
                let members = fm
                    .iter()
                    .cartesian_product(xm.iter())
                    .map(|(f, x)| self.vs_apply(f.clone(), x.clone()))
                    .collect();
                (cost, members)
            }
            VS::Abstraction(ref b) => {
                let (cost, members) = self.minimal_inhabitants_inner(b);
                let members = members
                    .iter()
                    .map(|b| self.vs_abstract(b.clone()))
                    .collect();
                (cost, members)
            }
            VS::Void | VS::Universe => unreachable!(),
            _ => {
                let mut members = HashSet::new();
                members.insert(VersionSpace::from(vs));
                (1f64, members)
            }
        };
        let v = (cost, Arc::new(members));
        let mut meta = self.all.get_mut(vs).unwrap();
        meta.inhabitants = Some(v.clone());
        v
    }
    fn minimal_function_inhabitants(&self, vs: &VersionSpace) -> (f64, Arc<HashSet<VersionSpace>>) {
        self.minimal_function_inhabitants_inner(&vs.0)
    }
    fn minimal_function_inhabitants_inner(
        &self,
        vs: &Arc<VS>,
    ) -> (f64, Arc<HashSet<VersionSpace>>) {
        if let Some(ref x) = self.all.get(vs).unwrap().function_inhabitants {
            return x.clone();
        }
        let (cost, members) = match **vs {
            VS::Union(ref u) => {
                let mut cost = f64::INFINITY;
                let mut members = HashSet::new();
                for child in u {
                    let (subcost, submembers) = self.minimal_inhabitants_inner(child);
                    match subcost.partial_cmp(&cost) {
                        Some(Ordering::Less) => {
                            cost = subcost;
                            members = (*submembers).clone();
                        }
                        Some(Ordering::Equal) => members.extend(submembers.iter().cloned()),
                        _ => (),
                    }
                }
                (cost, members)
            }
            VS::Application(ref f, ref x) => {
                let (fc, fm) = self.minimal_function_inhabitants_inner(f);
                let (xc, xm) = self.minimal_inhabitants_inner(x);
                let cost = fc + xc + EPSILON;
                let members = fm
                    .iter()
                    .cartesian_product(xm.iter())
                    .map(|(f, x)| self.vs_apply(f.clone(), x.clone()))
                    .collect();
                (cost, members)
            }
            VS::Abstraction(_) => (f64::NEG_INFINITY, HashSet::new()),
            VS::Void | VS::Universe => unreachable!(),
            _ => {
                let mut members = HashSet::new();
                members.insert(VersionSpace::from(vs));
                (1f64, members)
            }
        };
        let v = (cost, Arc::new(members));
        let mut meta = self.all.get_mut(vs).unwrap();
        meta.function_inhabitants = Some(v.clone());
        v
    }
    fn repeated_expansion(&self, vs: VersionSpace, n: u32) -> Vec<VersionSpace> {
        let mut spaces = vec![vs];
        for i in 0..n {
            let next = self.inversion(&spaces[i as usize]);
            spaces.push(next);
        }
        spaces
    }
    fn rewrite_reachable(
        &self,
        heads: Vec<VersionSpace>,
        n: u32,
    ) -> HashMap<Arc<VS>, Vec<VersionSpace>> {
        let mut vertices = HashSet::new();
        for vs in heads {
            vs.reachable(&mut vertices);
        }
        vertices
            .into_iter()
            .map(|vs| (vs.0.clone(), self.repeated_expansion(vs, n)))
            .collect()
    }
    pub fn super_version_space(&self, vs: &VersionSpace, n: u32) -> VersionSpace {
        if let Some(ref x) = self.all.get(&vs.0).unwrap().super_space {
            return x.clone();
        }
        let spaces = self.rewrite_reachable(vec![vs.clone()], n);
        let svs = self.super_version_space_inner(&vs.0, &spaces);
        let mut meta = self.all.get_mut(&vs.0).unwrap();
        meta.super_space = Some(svs.clone());
        svs
    }
    fn super_version_space_inner(
        &self,
        vs: &Arc<VS>,
        spaces: &HashMap<Arc<VS>, Vec<VersionSpace>>,
    ) -> VersionSpace {
        let mut components = spaces[vs].clone();
        components.push(VersionSpace::from(vs));
        match **vs {
            VS::Application(ref f, ref x) => {
                let f_ss = self.super_version_space_inner(f, spaces);
                let x_ss = self.super_version_space_inner(x, spaces);
                components.push(self.vs_apply(f_ss, x_ss))
            }
            VS::Abstraction(ref b) => {
                let b_ss = self.super_version_space_inner(b, spaces);
                components.push(self.vs_abstract(b_ss));
            }
            VS::Index(_) | VS::Terminal(_) => (),
            VS::Union(_) | VS::Void | VS::Universe => unreachable!(),
        }
        self.union(components)
    }
    fn incorporate_space(&self, vs: VS) -> VersionSpace {
        let vs = Arc::new(vs);
        if self.all.contains_key(&vs) {
            VersionSpace::from(self.all.get(&vs).unwrap().vs.upgrade().unwrap())
        } else {
            self.all.insert(Arc::clone(&vs), VersionMeta::new(&vs));
            VersionSpace(vs)
        }
    }
    fn rewrite(
        &self,
        e: &VersionSpace,
        ex: &lambda::Expression,
        v: VersionSpace,
        table: &mut HashMap<VersionSpace, RWCost>,
    ) -> RWCost {
        if table.contains_key(&v) {
            return table[&v].clone();
        }
        let r = if self.has_intersection(e, &v) {
            RWCost {
                f: Some((ex.clone(), 1.)),
                a: Some((ex.clone(), 1.)),
            }
        } else {
            match *v.0 {
                VS::Terminal(ref e) => RWCost {
                    f: Some((e.clone(), 1.)),
                    a: Some((e.clone(), 1.)),
                },
                VS::Index(i) => {
                    let e = Expression::Index(i);
                    RWCost {
                        f: Some((e.clone(), 1.)),
                        a: Some((e, 1.)),
                    }
                }
                VS::Application(ref f, ref x) => {
                    let f = self.rewrite(e, ex, VersionSpace::from(f), table);
                    let x = self.rewrite(e, ex, VersionSpace::from(x), table);
                    let cost = f.fc() + x.ac() + EPSILON;
                    if cost.is_finite() {
                        let f = f.f.unwrap().0;
                        let x = x.a.unwrap().0;
                        let ep = Expression::Application(Box::new(f), Box::new(x));
                        RWCost {
                            f: Some((ep.clone(), cost)),
                            a: Some((ep, cost)),
                        }
                    } else {
                        RWCost { f: None, a: None }
                    }
                }
                VS::Abstraction(ref b) => {
                    let b = self.rewrite(e, ex, VersionSpace::from(b), table);
                    let cost = b.ac() + EPSILON;
                    if cost.is_finite() {
                        let b = b.a.unwrap().0;
                        let ep = Expression::Abstraction(Box::new(b));
                        RWCost {
                            f: None,
                            a: Some((ep, cost)),
                        }
                    } else {
                        RWCost { f: None, a: None }
                    }
                }
                VS::Union(ref u) => {
                    let children: Vec<_> = u
                        .into_iter()
                        .map(|x| self.rewrite(e, ex, VersionSpace::from(x), table))
                        .collect();
                    let f_min = children
                        .iter()
                        .min_by(|a, b| a.fc().partial_cmp(&b.fc()).unwrap());
                    let a_min = children
                        .iter()
                        .min_by(|a, b| a.ac().partial_cmp(&b.ac()).unwrap());
                    RWCost {
                        f: f_min.and_then(|r| r.f.clone()),
                        a: a_min.and_then(|r| r.a.clone()),
                    }
                }
                VS::Void | VS::Universe => unreachable!(),
            }
        };
        table.insert(v, r.clone());
        r
    }
    fn rewrite_with_invention(
        &self,
        invented: &VersionSpace,
        vs: Vec<VersionSpace>,
    ) -> Vec<lambda::Expression> {
        let ex = invented.extract().pop().unwrap();
        let mut table = HashMap::new();
        let ret = vs
            .into_iter()
            .map(|v| self.rewrite(&invented, &ex, v, &mut table))
            .map(|r| r.a.unwrap().0)
            .collect();
        self.overlaps.clear();
        ret
    }
    fn best_inventions(
        &self,
        versions: &[Vec<VersionSpace>],
        beam_size: usize,
    ) -> Vec<VersionSpace> {
        let candidates: Vec<HashSet<VersionSpace>> = versions
            .iter()
            .map(|heads| {
                let mut out = HashSet::new();
                let mut vertices = HashSet::new();
                for head in heads {
                    head.reachable(&mut vertices);
                }
                for k in vertices {
                    out.extend(self.minimal_function_inhabitants(&k).1.iter().cloned());
                    out.extend(self.minimal_inhabitants(&k).1.iter().cloned());
                }
                out
            })
            .collect();
        let mut counts: HashMap<VersionSpace, usize> = HashMap::new();
        for ks in candidates {
            for k in ks {
                *counts.entry(k).or_default() += 1
            }
        }
        let candidates: HashSet<_> = counts
            .into_par_iter()
            .filter_map(|(vs, count)| {
                if count >= 2 && nontrivial(&vs.extract()[0]) {
                    Some(vs)
                } else {
                    None
                }
            })
            .collect();
        let beamer = beam::Beamer::new(self, &candidates, beam_size);
        let beam_table: CHashMap<VersionSpace, beam::Beam> = CHashMap::new();
        versions
            .into_par_iter()
            .flat_map(|heads| heads)
            .all(|head| {
                beamer.costs(&beam_table, head);
                true
            });
        let beam_table: HashMap<_, _> = beam_table.into_iter().collect();
        let mut vss: Vec<_> = beam_table
            .par_iter()
            .flat_map(|(_vs, beam)| beam.par_domain())
            .collect();
        vss.par_sort_unstable();
        vss.dedup();
        let mut pairs: Vec<_> = vss
            .into_par_iter()
            .map(|candidate| {
                let value = versions
                    .par_iter()
                    .map(|heads| {
                        heads
                            .into_iter()
                            .map(|head| {
                                let b = &beam_table[head];
                                b.cost(candidate).min(b.function_cost(candidate))
                            })
                            .fold(f64::INFINITY, f64::min)
                    })
                    .sum::<f64>();
                (candidate, value)
            })
            .collect();
        pairs.par_sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap());
        pairs
            .into_iter()
            .take(beam_size)
            .map(|(c, _)| c.clone())
            .collect()
    }
}

fn nontrivial(p: &Expression) -> bool {
    let mut indices = HashSet::new();
    let (mut primitives, mut collisions) = (0, 0);
    nontrivial_inner(p, &mut indices, &mut primitives, &mut collisions, 0);
    primitives > 1 || (primitives == 1 && collisions > 0)
}

/// returns (primitives, collisions)
fn nontrivial_inner(
    expr: &Expression,
    indices: &mut HashSet<usize>,
    p: &mut usize,
    c: &mut usize,
    depth: usize,
) {
    match expr {
        Expression::Application(ref f, ref x) => {
            nontrivial_inner(f, indices, p, c, depth);
            nontrivial_inner(x, indices, p, c, depth);
        }
        Expression::Abstraction(ref b) => nontrivial_inner(b, indices, p, c, depth + 1),
        Expression::Primitive(_) | Expression::Invented(_) => *p += 1,
        Expression::Index(i) if *i >= depth => {
            if !indices.insert(i - depth) {
                *c += 1
            }
        }
        Expression::Index(_) => (),
    }
}

fn free(expr: &lambda::Expression, depth: usize, s: &mut HashSet<usize>) {
    match expr {
        Expression::Application(f, x) => {
            free(f, depth, s);
            free(x, depth, s);
        }
        Expression::Abstraction(body) => free(body, depth + 1, s),
        Expression::Index(i) if *i >= depth => {
            s.insert(i - depth);
        }
        _ => (),
    }
}

fn close_invention(mut p: lambda::Expression) -> (lambda::Expression, HashMap<usize, usize>) {
    let mut s = HashSet::new();
    free(&p, 0, &mut s);
    let mapping = s
        .into_iter()
        .sorted()
        .into_iter()
        .enumerate()
        .map(|(j, fv)| (fv, j))
        .collect();
    close_invention_inner(&mut p, 0, &mapping);
    for _ in 0..mapping.len() {
        p = Expression::Abstraction(Box::new(p))
    }
    (p, mapping)
}

fn close_invention_inner(
    p: &mut lambda::Expression,
    depth: usize,
    mapping: &HashMap<usize, usize>,
) {
    match *p {
        Expression::Application(ref mut f, ref mut x) => {
            close_invention_inner(f, depth, mapping);
            close_invention_inner(x, depth, mapping);
        }
        Expression::Abstraction(ref mut body) => close_invention_inner(body, depth + 1, mapping),
        Expression::Index(ref mut i) if *i >= depth => {
            if let Some(k) = mapping.get(&(*i - depth)) {
                *i = k + depth;
            }
        }
        _ => (),
    }
}

fn rewrite(p: &lambda::Expression, inv: &lambda::Expression, e: &mut lambda::Expression) {
    if e == p {
        *e = inv.clone();
        return;
    }
    match *e {
        Expression::Application(ref mut f, ref mut x) => {
            rewrite(p, inv, f);
            rewrite(p, inv, x);
        }
        Expression::Abstraction(ref mut b) => {
            rewrite(p, inv, b);
        }
        _ => (),
    }
}

mod beam {
    use super::{VersionSpace, VersionTable, EPSILON, VS};
    use chashmap::CHashMap;
    use itertools::Itertools;
    use rayon::prelude::*;
    use std::collections::{HashMap, HashSet};

    pub struct Beamer<'a> {
        vt: &'a VersionTable,
        candidates: &'a HashSet<VersionSpace>,
        beam_size: usize,
    }
    impl<'a> Beamer<'a> {
        pub fn new(
            vt: &'a VersionTable,
            candidates: &'a HashSet<VersionSpace>,
            beam_size: usize,
        ) -> Beamer<'a> {
            Beamer {
                vt,
                candidates,
                beam_size,
            }
        }
        pub fn costs(&self, beam_table: &CHashMap<VersionSpace, Beam>, e: &VersionSpace) {
            if beam_table.contains_key(e) {
                return;
            }
            let mut beam = self.new_beam(e);
            match *e.0 {
                VS::Index(_) | VS::Terminal(_) => (),
                VS::Abstraction(ref b) => {
                    let b = &VersionSpace::from(b);
                    self.costs(beam_table, b);
                    let b = beam_table.get(b).unwrap();
                    for (ref i, c) in &b.relative_cost {
                        beam.relax(i, c + EPSILON);
                    }
                }
                VS::Application(ref f, ref x) => {
                    let f = &VersionSpace::from(f);
                    let x = &VersionSpace::from(x);
                    self.costs(beam_table, f);
                    self.costs(beam_table, x);
                    let f = beam_table.get(f).unwrap();
                    let x = beam_table.get(x).unwrap();
                    for i in f.function_domain().chain(x.domain()) {
                        let c = f.function_cost(i) + x.cost(i) + EPSILON;
                        beam.relax(i, c);
                        beam.relax_function(i, c);
                    }
                }
                VS::Union(ref u) => {
                    for z in u {
                        let z = &VersionSpace::from(z);
                        self.costs(beam_table, z);
                        let cz = beam_table.get(z).unwrap();
                        for (ref i, c) in &cz.relative_cost {
                            beam.relax(i, *c);
                        }
                        for (ref i, c) in &cz.relative_function_cost {
                            beam.relax_function(i, *c);
                        }
                    }
                }
                VS::Void | VS::Universe => unreachable!(),
            }
            beam.restrict();
            beam_table.insert(e.clone(), beam);
        }
        fn new_beam(&self, space: &VersionSpace) -> Beam {
            let meta = &self.vt.all.get(&space.0).unwrap();
            let (cost, inhabitants) = meta.inhabitants.clone().unwrap();
            let (function_cost, _) = meta.function_inhabitants.clone().unwrap();
            let default_costs = (cost, function_cost);
            let relative_cost: HashMap<_, _> = inhabitants
                .iter()
                .filter(|inhabitant| self.candidates.contains(inhabitant))
                .map(|vs| (vs.clone(), 1.))
                .collect();
            let relative_function_cost = relative_cost.clone();
            Beam {
                beam_size: self.beam_size,
                relative_cost,
                relative_function_cost,
                default_costs,
            }
        }
    }
    pub struct Beam {
        beam_size: usize,
        relative_cost: HashMap<VersionSpace, f64>,
        relative_function_cost: HashMap<VersionSpace, f64>,
        default_costs: (f64, f64),
    }
    impl Beam {
        pub fn par_domain(&self) -> impl ParallelIterator<Item = &VersionSpace> {
            self.relative_cost.par_iter().map(|(k, _v)| k)
        }
        pub fn domain(&self) -> impl Iterator<Item = &VersionSpace> {
            self.relative_cost.keys()
        }
        pub fn function_domain(&self) -> impl Iterator<Item = &VersionSpace> {
            self.relative_function_cost.keys()
        }
        fn restrict(&mut self) {
            if self.relative_cost.len() > self.beam_size {
                self.relative_cost = self
                    .relative_cost
                    .iter()
                    .sorted_by(|(_, ac), (_, bc)| ac.partial_cmp(&bc).unwrap())
                    .into_iter()
                    .take(self.beam_size)
                    .map(|(a, b)| (a.clone(), *b))
                    .collect();
            }
            if self.relative_function_cost.len() > self.beam_size {
                self.relative_function_cost = self
                    .relative_function_cost
                    .iter()
                    .sorted_by(|(_, ac), (_, bc)| ac.partial_cmp(&bc).unwrap())
                    .into_iter()
                    .take(self.beam_size)
                    .map(|(a, b)| (a.clone(), *b))
                    .collect();
            }
        }
        pub fn cost(&self, e: &VersionSpace) -> f64 {
            *self.relative_cost.get(e).unwrap_or(&self.default_costs.0)
        }
        pub fn function_cost(&self, e: &VersionSpace) -> f64 {
            *self
                .relative_function_cost
                .get(e)
                .unwrap_or(&self.default_costs.1)
        }
        fn relax(&mut self, e: &VersionSpace, cost: f64) {
            let c = self
                .relative_cost
                .entry(e.clone())
                .or_insert(self.default_costs.0);
            *c = f64::min(cost, *c);
        }
        fn relax_function(&mut self, e: &VersionSpace, cost: f64) {
            let c = self
                .relative_function_cost
                .entry(e.clone())
                .or_insert(self.default_costs.1);
            *c = f64::min(cost, *c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{induce_version_spaces, VersionTable};
    use programinduction::{lambda, ECFrontier, Task};
    use std::f64;

    fn noop_oracle(_: &lambda::Language, _: &lambda::Expression) -> f64 {
        f64::NEG_INFINITY
    }

    #[test]
    fn show_inversion() {
        let dsl = lambda::Language::uniform(vec![
            (">=", ptp!(@arrow[tp!(int), tp!(int), tp!(bool)])),
            ("+", ptp!(@arrow[tp!(int), tp!(int), tp!(int)])),
            ("0", ptp!(int)),
            ("1", ptp!(int)),
        ]);
        let e = dsl.parse("(+ +)").unwrap();
        {
            let mut vt = VersionTable::new();
            let vs = vt.incorporate(e.clone());
            let subs = vt.substitutions(&vs);
            for (k, v) in subs.iter() {
                eprintln!(
                    "substution k={}\tv={}",
                    k.0.display(&dsl),
                    v.0.display(&dsl)
                )
            }
            let inv = vt.inversion(&vs);
            for expr in inv.extract() {
                eprintln!("inversion: {}", dsl.display(&expr))
            }
        }
        let frontier = ECFrontier(vec![(e.clone(), 0., 0.)]);
        let task = Task {
            oracle: Box::new(noop_oracle),
            tp: ptp!(bool),
            observation: (),
        };
        let (dsl, _) = induce_version_spaces(
            &dsl,
            &lambda::CompressionParams::default(),
            &[task],
            vec![frontier],
            50,
        );
        for i in dsl.invented.len()..dsl.invented.len() {
            let &(ref expr, _, _) = &dsl.invented[i];
            eprintln!("invented {}", dsl.display(expr));
        }
    }
}
