#[feature(nll)]

use std::collections::{HashMap};//, HashSet, VecDeque};
//use std::rc::Rc;

#[derive(Hash, PartialEq, Eq, Clone)]
enum VersionSpace {
     Empty,
     Universe,
     Apply(usize, usize),
     Abstract(usize),
     Index(u32),
     Leaf(String),
     Union(Vec<usize>),
}

struct VersionTable {
       empty : usize,
       universe : usize,
       expressions : Vec<VersionSpace>,
       expression2index : HashMap<VersionSpace, usize>,
}

fn new_version_table() -> VersionTable {
    let mut t = VersionTable {
        empty: 0,
        universe: 1,
        expressions: Vec::new(),
        expression2index: HashMap::new(),
    };
    t.incorporate(VersionSpace::Empty);
    t.incorporate(VersionSpace::Universe);
    t
}

impl VersionTable {
    fn incorporate(&mut self, v : VersionSpace) -> usize {
        match self.expression2index.get(&v) {
            Some(i) => return *i,
            None => ()
        };
        
        let i = self.expressions.len();
        self.expressions.push(v.clone());
        self.expression2index.insert(v,i);
        i
    }

    fn union(&mut self, components : Vec<usize>) -> usize {
        for k in &components {
            if *k == self.universe {
                return self.universe;
            }
        }

        let mut collapsed_components : Vec<usize> = Vec::new();
        for k in &components {
            match & self.expressions[*k as usize] {
                VersionSpace::Union(subcomponents) => 
                    collapsed_components.extend(subcomponents),
                _ => {
                    if *k != self.empty {
                        collapsed_components.push(*k)
                    }
                }
            }
        };

        collapsed_components.sort();
        collapsed_components.dedup();
        
        if collapsed_components.len() == 0 {
            return self.empty;
        }

        if collapsed_components.len() == 1 {
            return collapsed_components[0];
        }


        self.incorporate(VersionSpace::Union(collapsed_components))
        
    }

    fn abstraction(&mut self, b : usize) -> usize {
        if b == self.empty { return b; }
        self.incorporate(VersionSpace::Abstract(b))
    }
    fn apply(&mut self, f : usize, x : usize) -> usize {
        if f == self.empty || x == self.empty { return self.empty; }
        self.incorporate(VersionSpace::Apply(f, x))
    }
    fn index(&mut self, i : u32) -> usize {
        self.incorporate(VersionSpace::Index(i))
    }

    fn shift_free(&mut self, j : usize, n : u32, c : u32) -> usize {
        if n == 0 {
            return j;
        }

        match &self.expressions[j] {
            &VersionSpace::Index(i) => {
                if i < c { return j; }
                if i >= n + c { return self.index(i - n); }
                self.empty
            },
            &VersionSpace::Abstract(b) => {
                let body = self.shift_free(b, n, c + 1);
                self.abstraction(body)
            },
            &VersionSpace::Apply(f,x) => {
                let argument = self.shift_free(x,n,c);
                let function = self.shift_free(f,n,c);
                self.apply(function, argument)
            },
            &VersionSpace::Union(ref u) => {
                let mut new_union = Vec::new();
                for e in u {
                    new_union.push(self.shift_free(*e,n,c));
                };
                self.union(new_union)
            },
            _ => j
        }
    }

    fn intersection(&mut self, a : usize, b : usize) -> usize {
        if a == self.empty || b == self.empty {
            return self.empty;
        }
        if a == self.universe {
            return b;
        }
        if b == self.universe {
            return a;
        }
        if a == b {
            return a;
        }

        match (self.expressions[a], self.expressions[b]) {
            (VersionSpace::Abstract(b1), VersionSpace::Abstract(b2)) => {
                let body = self.intersection(b1,b2);
                self.abstraction(body)
            },
            (VersionSpace::Apply(f1,x1),VersionSpace::Apply(f2,x2)) => {
                let f = self.intersection(f1,f2);
                let x = self.intersection(x1,x2);
                self.apply(f,x)
            },
            (VersionSpace::Union(u1),VersionSpace::Union(u2)) => {
                let mut new_union = Vec::new();
                for x in &u1 {
                    for y in &u2 {
                        new_union.push(self.intersection(*x,*y));
                    }
                }
                self.union(new_union)
            },
            (VersionSpace::Union(u),_) => {
                let mut new_union = Vec::new();
                for y in &u {
                    new_union.push(self.intersection(b,*y));
                }
                self.union(new_union)
            },
            (_,VersionSpace::Union(u)) => {
                let mut new_union = Vec::new();
                for y in &u {
                    new_union.push(self.intersection(a,*y));
                }
                self.union(new_union)
            },
            _ => self.empty
        }                
    }
                
                    
}

fn main() {
    println!("hello world");

    let mut t = new_version_table();

    println!("{}",(t.incorporate(VersionSpace::Empty)));
    println!("{}",(t.incorporate(VersionSpace::Universe)));
    println!("{}",(t.incorporate(VersionSpace::Index(0))));
    println!("{}",(t.incorporate(VersionSpace::Index(0))));
    let i0 = t.index(0);
    let i1 = t.index(1);
    println!("{}",(t.union(vec![i0,i1])));
    let i0_ = t.index(0);
    // let i1_ = t.index(1);
    println!("{}",(t.union(vec![i0_,i0_])));
    
}
