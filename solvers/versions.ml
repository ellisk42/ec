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
                let a = version_apply t f x in
                let v = intersection t v1 v2 in
                if v <> t.void then begin
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

let _ =
  let t = new_version_table() in
  let p = parse_program "(lambda (fold $0 empty (lambda (lambda (cons (+ (+ 5 5) (+ $1 $1)) $0)))))" |> get_some in
  let p' = parse_program "(+ 9 9)" |> get_some in
  let j = time_it "calculated versions base" (fun () -> p |> incorporate t |> recursive_inversion t |> recursive_inversion t  |> recursive_inversion t) in
  extract t j |> List.map ~f:(fun r ->
      (* Printf.printf "%s\n\t%s\n" (string_of_program r) *)
      (*   (beta_normal_form r |> string_of_program); *)
      (* flush_everything(); *)
      assert ((string_of_program p) = (beta_normal_form r |> string_of_program)));
  Printf.printf "Enumerated %d version spaces.\n"
    (t.i2s.ra_occupancy)
