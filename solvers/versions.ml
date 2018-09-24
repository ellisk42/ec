open Core
open Program
open Utils
open Type




type vs =
  | Union of (int list)
  | ApplySpace of int*int
  | AbstractSpace of int
  | IndexSpace of int
  | TerminalSpace of program
  | Universe | Void

type vt = {universe : int;
           void : int;
           s2i : (vs,int) Hashtbl.t;
           i2s : vs ra;
           (* equivalence classes *)
           (* equivalence_class : int ra; *)
           (* dynamic programming *)
           recursive_inversion_table : (int option) ra;
           substitution_table : ((int*int), ((int,int) Hashtbl.t)) Hashtbl.t;}

let index_table t index = get_resizable t.i2s index
let version_table_size t = t.i2s.ra_occupancy

let incorporate_space t v : int =
  match Hashtbl.find t.s2i v with
  | Some(i) -> i
  | None -> begin
      let i = t.i2s.ra_occupancy in
      Hashtbl.set t.s2i ~key:v ~data:i;
      push_resizable t.i2s v;
      push_resizable t.recursive_inversion_table None;
      (* push_resizable t.equivalence_class (union_find_node i); *)
      i
    end

let new_version_table() : vt =
  let t = {void=0;
           universe=1;
           s2i=Hashtbl.Poly.create();
           i2s=empty_resizable();
           (* equivalence_class=empty_resizable(); *)
           substitution_table=Hashtbl.Poly.create();
           recursive_inversion_table=empty_resizable();} in
  assert (incorporate_space t Void = t.void);
  assert (incorporate_space t Universe = t.universe);
  t

let version_apply t f x =
  if f = t.void || x = t.void then t.void else incorporate_space t (ApplySpace(f,x))
let version_abstract t b =
  if b = t.void then t.void else incorporate_space t (AbstractSpace(b))
let version_index t i = incorporate_space t (IndexSpace(i))
let version_terminal t e =
  incorporate_space t (TerminalSpace(e))
let union t vs =
  if List.mem vs (t.universe) ~equal:(=) then t.universe else
    let vs = vs |> List.concat_map  ~f:(fun v -> match index_table t v with
        | Union(stuff) -> stuff
        | Void -> []
        | Universe -> assert false
        | _ -> [v]) |> List.dedup_and_sort ~compare:(-) in
    match vs with
    | [] -> t.void
    | [v] -> v
    | _ -> incorporate_space t (Union(vs))
    
let rec incorporate t e =
  match e with
  | Index(i) -> version_index t i
  | Abstraction(b) -> version_abstract t (incorporate t b)
  | Apply(f,x) -> version_apply t (incorporate t f) (incorporate t x)
  | Primitive(_,_,_) | Invented(_,_) -> version_terminal t (strip_primitives e)


let rec extract t j =
  match index_table t j with
  | Union(u) -> List.concat_map u ~f:(extract t)
  | ApplySpace(f,x) ->
    extract t f |> List.concat_map ~f:(fun f' ->
        extract t x |> List.map ~f:(fun x' ->
            Apply(f',x')))
  | IndexSpace(i) -> [Index(i)]
  | Void -> []
  | TerminalSpace(p) -> [p]
  | AbstractSpace(b) ->
    extract t b |> List.map ~f:(fun b' ->
        Abstraction(b'))
  | Universe -> [primitive "UNIVERSE" t0 ()]

let rec child_spaces t j =
  (j :: 
   match index_table t j with
   | Union(u) -> List.map u ~f:(child_spaces t) |> List.concat
   | ApplySpace(f,x) -> child_spaces t f @ child_spaces t x
   | AbstractSpace(b) -> child_spaces t b
   | _ -> [])
  |> List.dedup_and_sort ~compare:(-)
    
let rec shift_free ?c:(c=0) t ~n ~index =
  if n = 0 then index else
    match index_table t index with
    | Union(indices) ->
      union t (indices |> List.map ~f:(fun i -> shift_free ~c:c t ~n:n ~index:i))
    | IndexSpace(i) when i < c -> index (* below cut off - bound variable *)
    | IndexSpace(i) when i >= n + c -> version_index t (i - n) (* free variable *)
    | IndexSpace(_) -> t.void
    | ApplySpace(f,x) ->
      version_apply t (shift_free ~c:c t ~n:n ~index:f) (shift_free ~c:c t ~n:n ~index:x)
    | AbstractSpace(b) ->
      version_abstract t (shift_free ~c:(c+1) t ~n:n ~index:b)
    | TerminalSpace(_) | Universe | Void -> index

let rec intersection t a b =
  match index_table t a, index_table t b with
  | Universe, _ -> b
  | _, Universe -> a
  | Void, _ | _, Void -> t.void
  | Union(xs), Union(ys) ->
    xs |> List.concat_map ~f:(fun x -> ys |> List.map ~f:(fun y -> intersection t x y)) |> union t
  | Union(xs), _ -> 
    xs |> List.map ~f:(fun x -> intersection t x b) |> union t
  | _, Union(xs) ->
    xs |> List.map ~f:(fun x -> intersection t x a) |> union t
  | AbstractSpace(b1), AbstractSpace(b2) ->
    version_abstract t (intersection t b1 b2)
  | ApplySpace(f1,x1), ApplySpace(f2,x2) ->
    version_apply t (intersection t f1 f2) (intersection t x1 x2)
  | IndexSpace(i1), IndexSpace(i2) when i1 = i2 -> a
  | TerminalSpace(t1), TerminalSpace(t2) when t1 = t2 -> a
  | _ -> t.void

let rec have_intersection t a b =
  match index_table t a, index_table t b with
  | Void, _ | _, Void -> false
  | Universe, _ -> true
  | _, Universe -> true
  | Union(xs), Union(ys) ->
    xs |> List.exists ~f:(fun x -> ys |> List.exists ~f:(fun y -> have_intersection t x y))
  | Union(xs), _ -> 
    xs |> List.exists ~f:(fun x -> have_intersection t x b)
  | _, Union(xs) ->
    xs |> List.exists ~f:(fun x -> have_intersection t x a)
  | AbstractSpace(b1), AbstractSpace(b2) ->
    have_intersection t b1 b2
  | ApplySpace(f1,x1), ApplySpace(f2,x2) ->
    have_intersection t f1 f2 && have_intersection t x1 x2
  | IndexSpace(i1), IndexSpace(i2) when i1 = i2 -> true
  | TerminalSpace(t1), TerminalSpace(t2) when t1 = t2 -> true
  | _ -> false

let rec substitutions t ?n:(n=0) index =
  match Hashtbl.find t.substitution_table (index,n) with
  | Some(s) -> s
  | None ->

    let s = shift_free t ~n:n ~index in
    let m = Hashtbl.Poly.create() in
    if s <> t.void then ignore(Hashtbl.add m ~key:s ~data:(version_index t n));

    begin 
      match index_table t index with
      | TerminalSpace(_) -> ignore(Hashtbl.add m ~key:t.universe ~data:index)
      | IndexSpace(i) ->
        ignore(Hashtbl.add m ~key:t.universe ~data:(if i < n then index else version_index t (1+i)))
      | AbstractSpace(b) ->
        substitutions t ~n:(n+1) b |> Hashtbl.iteri ~f:(fun ~key ~data ->
            Hashtbl.add_exn m ~key ~data:(version_abstract t data))
      | Union(u) ->
        let new_mapping = Hashtbl.Poly.create() in
        u |> List.iter ~f:(fun x ->
            substitutions t ~n x |> Hashtbl.iteri ~f:(fun ~key:v ~data:b ->
                match Hashtbl.find new_mapping v with
                | Some(stuff) -> Hashtbl.set new_mapping ~key:v ~data:(b :: stuff)
                | None -> Hashtbl.set new_mapping ~key:v ~data:[b]));
        new_mapping |> Hashtbl.iteri ~f:(fun ~key ~data ->
            Hashtbl.set m ~key ~data:(union t data))          

      | ApplySpace(f, x) ->
        let new_mapping = Hashtbl.Poly.create() in
        let fm = substitutions t ~n f in
        let xm = substitutions t ~n x in

        fm |> Hashtbl.iteri ~f:(fun ~key:v1 ~data:f ->
            xm |> Hashtbl.iteri ~f:(fun ~key:v2 ~data:x ->
                if have_intersection t v1 v2 then begin
                  let v = intersection t v1 v2 in
                  let a = version_apply t f x in
                  match Hashtbl.find new_mapping v with
                  | Some(stuff) -> Hashtbl.set new_mapping ~key:v ~data:(a :: stuff)
                  | None -> Hashtbl.set new_mapping ~key:v ~data:[a]
                end));

        new_mapping |> Hashtbl.iteri ~f:(fun ~key ~data ->
            Hashtbl.set m ~key ~data:(union t data))          

      | _ -> ()
    end;
    Hashtbl.set (t.substitution_table) ~key:(index,n) ~data:m;
    m

let inversion t j =
  substitutions t j |> Hashtbl.to_alist |>
  List.filter_map ~f:(fun (v,b) ->
      if v = t.universe || index_table t b = IndexSpace(0) then None else 
      Some(version_apply t (version_abstract t b) v)) |>
  union t

let rec recursive_inversion t j =
  match get_resizable t.recursive_inversion_table j with
  | Some(ri) -> ri
  | None ->
    let ri = 
      match index_table t j with
      | Union(u) -> union t (u |> List.map ~f:(recursive_inversion t))
      | _ ->
        let top_inversions = substitutions t j |> Hashtbl.to_alist |>
                             List.filter_map ~f:(fun (v,b) ->
                                 if v = t.universe || index_table t b = IndexSpace(0) then None else 
                                   Some(version_apply t (version_abstract t b) v))
        in
        let child_inversions = match index_table t j with
          | ApplySpace(f, x) -> [version_apply t (recursive_inversion t f) x;
                                 version_apply t f (recursive_inversion t x)]
          | AbstractSpace(b) -> [version_abstract t (recursive_inversion t b)]
          | _ -> []
        in
        union t (child_inversions @ top_inversions)
    in
    set_resizable t.recursive_inversion_table j (Some(ri));
    ri

let beta_pruning t j = 
let rec beta_pruning' ?isApplied:(isApplied=false) ?canBeta:(canBeta=true)
    t j =
  match index_table t j with
  | ApplySpace(f,x) ->
    let f' =  beta_pruning' ~canBeta ~isApplied:true t f in
    let x' =  beta_pruning' ~canBeta ~isApplied:false t x in
    version_apply t f' x'
  | AbstractSpace(_) when isApplied && not canBeta -> t.void
  | AbstractSpace(b) when isApplied && canBeta ->
    let b' = beta_pruning' ~isApplied:false ~canBeta:false t b in
    version_abstract t b'
  | AbstractSpace(b) ->
    let b' = beta_pruning' ~isApplied:false ~canBeta t b in
    version_abstract t b'
  | Union(u) ->
    u |> List.map ~f:(beta_pruning' ~isApplied ~canBeta t) |> union t
  | IndexSpace(_) | TerminalSpace(_) | Universe | Void -> j
in beta_pruning' t j
  
let rec log_version_size t j = match index_table t j with
  | ApplySpace(f,x) -> log_version_size t f +. log_version_size t x
  | AbstractSpace(b) -> log_version_size t b
  | Union(u) -> u |> List.map ~f:(log_version_size t) |> lse_list
  | _ -> 0.

let rec n_step_inversion t ~n j =

  (* list of length (n+1), corresponding to 0 steps, 1, ..., n *)
  let rec n_step ?completed:(completed=0) current : int list =
    if completed = n then [current] else
      let next = recursive_inversion t current in
      let next = beta_pruning t next in
      current :: n_step ~completed:(completed+1) next
  in

  let rec visit j =
    let children = match index_table t j with
        | Union(_) | Void | Universe -> assert false
        | ApplySpace(f,x) -> version_apply t (visit f) (visit x)
        | AbstractSpace(b) -> version_abstract t (visit b)
        | IndexSpace(_) | TerminalSpace(_) -> j
    in 
    union t (children :: n_step j)
  in 
    
  visit j


let reachable_versions t indices : int list =
  let visited = Hash_set.Poly.create() in

  let rec visit j = if Hash_set.mem visited j then () else
      (Hash_set.add visited j;
       match index_table t j with
       | Universe | Void | IndexSpace(_) | TerminalSpace(_) -> ()
       | AbstractSpace(b) -> visit b
       | ApplySpace(f,x) -> (visit f; visit x)
       | Union(u) -> u |> List.iter ~f:visit)
  in
  indices |> List.iter ~f:visit;
  Hash_set.fold visited ~f:(fun a x -> x :: a) ~init:[]
  

(* cost calculations *)
let epsilon_cost = 0.01;;

(* Holds the minimum cost of each version space *)
type cost_table = {function_cost : ((float*(int list)) option) ra;
                   argument_cost : ((float*(int list)) option) ra;
                   cost_table_parent : vt;}

let empty_cost_table t = {function_cost = empty_resizable();
                          argument_cost = empty_resizable();
                          cost_table_parent = t;}
let rec minimum_cost_inhabitants ?given:(given=None) ?canBeLambda:(canBeLambda=true) t j : float*(int list) =
  let caching_table = if canBeLambda then t.argument_cost else t.function_cost in
  ensure_resizable_length caching_table (j + 1) None;
  
  match get_resizable caching_table j with
  | Some(c) -> c
  | None ->
    let c =
      match given with
      | Some(invention) when have_intersection t.cost_table_parent invention j -> (1., [invention])
      | _ -> 
      match index_table t.cost_table_parent j with
      | Universe | Void -> assert false
      | IndexSpace(_) | TerminalSpace(_) -> (1., [j])
      | Union(u) ->
        let children = u |> List.map ~f:(minimum_cost_inhabitants ~given ~canBeLambda t) in
        minimum_by fst children
        (* let c = children |> List.map ~f:(fun (cost,_) -> cost) |> fold1 min in *)
        (* if is_invalid c then (c,[]) else  *)
        (*   let children = children |> List.filter ~f:(fun (cost,_) -> cost = c) in *)
        (*   (c, children |> List.concat_map ~f:(fun (_,p) -> p)) *)
      | AbstractSpace(b) when canBeLambda ->
        let cost, children = minimum_cost_inhabitants ~given ~canBeLambda:true t b in
        (cost+.epsilon_cost, children |> List.map ~f:(version_abstract t.cost_table_parent))
      | AbstractSpace(b) -> (Float.infinity,[])
      | ApplySpace(f,x) ->
        let fc, fs = minimum_cost_inhabitants ~given ~canBeLambda:false t f in
        let xc, xs = minimum_cost_inhabitants ~given ~canBeLambda:true t x in
        if is_invalid fc || is_invalid xc then (Float.infinity,[]) else
          (fc+.xc+.epsilon_cost,
          fs |> List.map ~f:(fun f' -> xs |> List.map ~f:(fun x' -> version_apply t.cost_table_parent f' x')) |> List.concat)
    in
    let cost, indices = c in
    let indices = indices |> List.dedup_and_sort ~compare:(-) in
    let c = (cost, indices) in
    set_resizable caching_table j (Some(c));
    c

type beam = {default_function_cost : float;
             default_argument_cost : float;
             mutable relative_function : (int,float) Hashtbl.t;
             mutable relative_argument : (int,float) Hashtbl.t;}

let narrow ~bs b =
  let narrow bm =
    if Hashtbl.length bm > bs then 
      let sorted = Hashtbl.to_alist bm |> List.sort ~compare:(fun (_,c1) (_,c2) -> Float.compare c1 c2) in      
      Hashtbl.Poly.of_alist_exn (List.take sorted bs)          
    else bm
  in
  b.relative_function <- narrow b.relative_function;
  b.relative_argument <- narrow b.relative_argument
;;
let relax table key data =
  match Hashtbl.find table key with
  | None -> Hashtbl.set table ~key ~data
  | Some(old) when old > data -> Hashtbl.set table ~key ~data
  | Some(_) -> ()
;;
let relative_function b i = match Hashtbl.find b.relative_function i with
  | None -> b.default_function_cost
  | Some(c) -> c
;;
let relative_argument b i = match Hashtbl.find b.relative_argument i with
  | None -> b.default_argument_cost
  | Some(c) -> c
;;

  
let beam_costs ~ct ~bs (candidates : int list) (frontier_indices : (int list) list)
  : (float*int) list =
  let candidates' = candidates in
  let candidates = Hash_set.Poly.of_list candidates in
  let caching_table = empty_resizable() in
  let v = ct.cost_table_parent in

  let rec calculate_costs j =
    ensure_resizable_length caching_table (j + 1) None;    
    match get_resizable caching_table j with
    | Some(bm) -> bm
    | None ->
      let default_argument_cost, inhabitants = minimum_cost_inhabitants ~canBeLambda:true ct j in
      let default_function_cost, _ = minimum_cost_inhabitants ~canBeLambda:false ct j in
      let bm = {default_argument_cost;
                default_function_cost;
                relative_function=Hashtbl.Poly.create();
                relative_argument=Hashtbl.Poly.create();}
      in
      inhabitants |> List.filter ~f:(Hash_set.mem candidates) |> List.iter ~f:(fun candidate ->
          Hashtbl.set bm.relative_function ~key:candidate ~data:1.;
          Hashtbl.set bm.relative_argument ~key:candidate ~data:1.);
      (match index_table v j with
       | AbstractSpace(b) ->
         let child = calculate_costs b in
         child.relative_argument |> Hashtbl.iteri ~f:(fun ~key ~data ->
             relax bm.relative_argument key (data+.epsilon_cost))
       | ApplySpace(f,x) ->
         let fb = calculate_costs f in
         let xb = calculate_costs x in
         let domain = Hashtbl.keys fb.relative_function @ Hashtbl.keys xb.relative_argument in
         domain |> List.iter ~f:(fun i ->
             let c = epsilon_cost +. relative_function fb i +. relative_argument xb i in
             relax bm.relative_function i c;
             relax bm.relative_argument i c)
       | Union(u) -> u |> List.iter ~f:(fun u ->
           let child = calculate_costs u in
           child.relative_function |> Hashtbl.iteri ~f:(fun ~key ~data ->
               relax bm.relative_function key data);
           child.relative_argument |> Hashtbl.iteri ~f:(fun ~key ~data ->
               relax bm.relative_argument key data))
       | IndexSpace(_) | Universe | Void | TerminalSpace(_) -> ());
      narrow ~bs bm;
      set_resizable caching_table j (Some(bm));
      bm
  in

  let frontier_beams = frontier_indices |> List.map ~f:(List.map ~f:calculate_costs) in

  let score i =
    let invention_size, _ = minimum_cost_inhabitants ct i in
    let corpus_size = frontier_beams |> List.map ~f:(fun bs ->
        bs |> List.map ~f:(fun b -> min (relative_argument b i) (relative_function b i)) |>
        fold1 min) |> fold1 (+.)
    in
    invention_size +. corpus_size
  in

  let scored = candidates' |> List.map ~f:(fun i -> (score i,i)) in
  scored |> List.sort ~compare:(fun (s1,_) (s2,_) -> Float.compare s1 s2)

  
  
