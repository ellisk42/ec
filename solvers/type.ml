open Core
open Funarray

type tp = 
  | TID of int
  | TCon of string * tp list * bool
            

let is_polymorphic = function
  | TID(_) -> true
  | TCon(_,_,p) -> p

let rec tp_eq a b =
  match (a,b) with
  | (TID(x),TID(y)) -> x = y
  | (TCon(k1,as1,_),TCon(k2,as2,_)) ->
    k1 = k2 && (type_arguments_equal as1 as2)
  | _ -> false
and type_arguments_equal xs ys =
  match (xs,ys) with
  | (a :: b, c :: d) -> tp_eq a c && type_arguments_equal b d
  | ([],[]) -> true
  | _ -> false

let kind n ts = TCon(n,ts,ts |> List.exists ~f:is_polymorphic)

(* A context has a size (fst) as well as an array mapping indices to types (snd; the substitution)
The substitution is stored in reverse order
 *)
type tContext = int * ((tp option) Funarray.funarray);;
let empty_context : tContext = (0, Funarray.empty);;

let make_arrow t q = kind "->" [t;q];;
let (@>) = make_arrow;;
let is_arrow = function
  | TCon("->",_,_) -> true
  | _ -> false;;

(* arguments_and_return_up_type (t1 @> t2 @> ... @> T) = ([t1;t2;...] T) *)
let rec arguments_and_return_of_type t =
  match t with
  | TCon("->",[p;q],_) ->
    let (arguments,return) = arguments_and_return_of_type q in
    (p::arguments,return)
  | _ -> ([],t)

(* return_of_type (t1 @> t2 @> ... @> T) = T *)
let rec return_of_type t =
  match t with
  | TCon("->",[_;q],_) -> return_of_type q 
  | _ -> t

(* arguments_of_type (t1 @> t2 @> ... @> T) = [t1;t2;...] *)
let rec arguments_of_type t =
  match t with
  | TCon("->",[p;q],_) -> p :: arguments_of_type q
  | _ -> []

let right_of_arrow t =
  match t with
  | TCon("->",[_;p],_) -> p
  | _ -> raise (Failure "right_of_arrow")

let left_of_arrow t =
  match t with
  | TCon("->",[p;_],_) -> p
  | _ -> raise (Failure "right_of_arrow")


let rec show_type (is_return : bool) (t : tp) : string = 
  match t with
  | TID(i) -> "t"^string_of_int i
  | TCon(k,[],_) -> k
  | TCon(k,[p;q],_) when k = "->" ->
    if is_return then
      (show_type false p)^" -> "^(show_type true q)
    else
      "("^(show_type false p)^" -> "^(show_type true q)^")"
  | TCon(k,a,_) -> k^"("^(String.concat ~sep:", " (List.map a ~f:(show_type true)))^")"
                 
let string_of_type = show_type true

let makeTID (next, substitution) =  
  (TID(next), (next + 1, Funarray.cons None substitution))

let rec makeTIDs (n : int) (k : tContext) : tContext =
  if n = 0 then k else makeTIDs (n-1) (makeTID k |> snd)

let bindTID i t (next, bindings) : tContext = 
  (next, Funarray.update bindings (next - i - 1) (Some(t)))

let lookupTID (next, bindings) j =
  assert (j < next);
  Funarray.lookup bindings (next - j - 1)

(* let rec chaseType (context : tContext) (t : tp) : tp*tContext =  *)
(*   match t with *)
(*   | TCon(s, ts) -> *)
(*     let (ts_, context_) = *)
(*       List.fold_right ts  *)
(* 	~f:(fun t (tz, k) -> *)
(* 	    let (t_, k_) = chaseType k t in *)
(* 	    (t_ :: tz, k_)) ~init:([], context) *)
(*     in (TCon(s, ts_), context_) *)
(*   | TID(i) ->  *)
(*     match TypeMap.find (snd context) i with *)
(*     | Some(hit) ->  *)
(*       let (t_, context_) = chaseType context hit in *)
(*       let substitution = TypeMap.add (snd context_) i t_ in *)
(*       (t_, (fst context_, substitution)) *)
(*     | None -> (t,context) *)

let rec applyContext k t =
  if not (is_polymorphic t) then (k,t) else 
  match t with
    | TCon(c,xs,_) ->
      let (k,xs) = List.fold_right xs ~init:(k,[]) ~f:(fun x (k,xs) ->
          let (k,x) = applyContext k x in
          (k,x :: xs)) in
      (k, kind c xs)
  | TID(j) ->
    match lookupTID k j with
    | None -> (k,t)
    | Some(tp) ->
      let (k,tp') = applyContext k tp in
      let k = if tp_eq tp tp' then k else bindTID j tp' k in
      (k,tp')



let rec occurs (i : int) (t : tp) : bool =
  if not (is_polymorphic t) then false else 
  match t with
  | TID(j) -> j = i
  | TCon(_,ts,_) -> 
    List.exists ts (occurs i)

let occursCheck = true

exception UnificationFailure

let rec might_unify t1 t2 = 
  match (t1, t2) with
  | (TCon(k1,as1,_), TCon(k2,as2,_)) when k1 = k2 -> 
    List.for_all2_exn as1 as2 might_unify
  | (TID(_),_) -> true
  | (_,TID(_)) -> true
  | _ -> false

let rec unify context t1 t2 : tContext =
  let (context,t1) = applyContext context t1 in
  let (context,t2) = applyContext context t2 in
  if (not (is_polymorphic t1)) && (not (is_polymorphic t2))
  then begin
    if tp_eq t1 t2 then context else raise UnificationFailure
  end
  else 
    match (t1,t2) with
    | (TID(j),t) ->
      if tp_eq t1 t2 then context 
      else (if occurs j t then raise UnificationFailure else bindTID j t context)
    | (t,TID(j)) ->
      if t1 = t2 then context 
      else (if occurs j t then raise UnificationFailure else bindTID j t context)
    | (TCon(k1,as1,_),TCon(k2,as2,_)) when k1 = k2 ->
      List.fold2_exn ~init:context as1 as2 ~f:unify
    | _ -> raise UnificationFailure

let instantiate_type k t = 
  let substitution = ref [] in
  let k = ref k in
  let rec instantiate j =
    if not (is_polymorphic j) then j else 
    match j with
      | TID(i) ->
        (try List.Assoc.find_exn ~equal:(fun a b -> a = b) !substitution i
	 with Not_found->
    let (t,k') = makeTID !k in
    k := k';
    substitution := (i,t)::!substitution;
    t)
    | TCon(k,js,_) -> kind k (List.map ~f:instantiate js)
  in let q = instantiate t in
  (!k, q)


let applyContext' k t =
  let new_context, t' = applyContext !k t in
  k := new_context;
  t';;
let unify' context_reference t1 t2 = context_reference := unify !context_reference t1 t2;;
let instantiate_type' context_reference t =
  let new_context, t' = instantiate_type !context_reference t in
  context_reference := new_context;
  t'
  

(* puts a type into normal form *)
let canonical_type t = 
  let next = ref 0 in
  let substitution = ref [] in
  let rec canon q = 
    match q with
    | TID(i) -> (try TID(List.Assoc.find_exn ~equal:(=) !substitution i)
		 with Not_found ->
                   substitution := (i,!next)::!substitution; next := (1+ !next); TID(!next-1))
    | TCon(k,a,_) -> kind k (List.map ~f:canon a)
  in canon t

let rec next_type_variable t = 
  match t with
  | TID(i) -> i+1
  | TCon(_,[],_) -> 0
  | TCon(_,is,_) -> List.fold_left ~f:max ~init:0 (List.map is next_type_variable)

(* tries to instantiate a universally quantified type with a given request *)
(* let instantiated_type universal_type requested_type = 
 *   try
 *     let (universal_type,c) = instantiate_type empty_context universal_type in
 *     let (requested_type,c) = instantiate_type c requested_type in
 *     let c = unify c universal_type requested_type in
 *     Some(canonical_type (applyContext c universal_type |> snd))
 *   with _ -> None *)

(* let compile_unifier t =
 *   let t = canonical_type t in
 *   let (xs,r) = arguments_and_return_of_type t in
 *   let free_variables = next_type_variable  in
 *   
 * 
 *   fun (target, context) ->
 *     if not (might_unify target r) then raise UnificationFailure else
 *       let bindings = Array.make free_variables None in
 * 
 *       let rec u k template original =
 *         match (template, original) with
 *         | (TID(templateVariable), v) -> begin
 *             match bindings.(templateVariable) with
 *             | Some(bound) -> unify k bound v
 *             | None -> begin
 *                 bindings.(templateVariable) <- v;
 *                 context
 *               end
 *           end
 *         | () *)



let rec get_arity = function
  | TCon(a,[_;r],_) when a = "->" -> 1+get_arity r
  | _ -> 0

let rec pad_type_with_arguments context n t =
  if n = 0 then (context,t) else
    let (a,context) = makeTID context in
    let (context,suffix) = pad_type_with_arguments context (n - 1) t in
    (context, a @> suffix)


let make_ground g = TCon(g,[],false);;


let tint = make_ground "int";;
let tcharacter = make_ground "char";;
let treal = make_ground "real";;
let tboolean = make_ground "bool";;
let turtle = make_ground "turtle";;
let ttower = make_ground "tower";;
let tstate = make_ground "tstate";;
let tscalar = make_ground "tscalar";;
let tangle = make_ground "tangle";;
let tlength = make_ground "tlength";;
let t0 = TID(0);;
let t1 = TID(1);;
let t2 = TID(2);;
let t3 = TID(3);;
let t4 = TID(4);;
let tlist t = kind "list" [t];;
let tstring = tlist tcharacter;;
let tvar               = make_ground "var"
let tprogram           = make_ground "program"
let tmaybe t           = kind "maybe" [t]
let tcanvas            = tlist tint

(** CLEVR Types -- types.ml **)
let tclevrcolor = make_ground "tclevrcolor";;
let tclevrsize = make_ground "tclevrsize";;
let tclevrmaterial = make_ground "tclevrmaterial";;
let tclevrshape = make_ground "tclevrshape";;
let tclevrrelation = make_ground "tclevrrelation";;
let tclevrobject = make_ground "tclevrobject";;

let unify_many_types ts =
  let k = empty_context in
  let (t,k) = makeTID k in
  let k = ref k in
  ts |> List.iter ~f:(fun t' ->
      let (k',t') = instantiate_type !k t' in
      k := unify k' t' t);
  applyContext !k t |> snd
     

let rec deserialize_type j =
  let open Yojson.Basic.Util in
  try
    let k = j |> member "constructor" |> to_string in
    let a = j |> member "arguments" |> to_list |> List.map ~f:deserialize_type in
    kind k a
  with _ ->
    let i = j |> member "index" |> to_int in
    TID(i)


let rec serialize_type t =
  let open Yojson.Basic in
  let j : json =
  match t with
  | TID(i) -> `Assoc(["index",`Int(i)])
  | TCon(k,a,_) ->
    `Assoc(["constructor",`String(k);
            "arguments",`List(a |> List.map ~f:serialize_type)])
  in
  j
