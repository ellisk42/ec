open Core


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

type tContext = int * (int*tp) list
let empty_context = (0,[]);;

let make_arrow t q = kind "->" [t;q];;
let (@>) = make_arrow;;

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

let rec show_type (is_return : bool) (t : tp) : string = 
  match t with
  | TID(i) -> "TV["^string_of_int i^"]"
  | TCon(k,[],_) -> k
  | TCon(k,[p;q],_) when k = "->" ->
    if is_return then
      (show_type false p)^" -> "^(show_type true q)
    else
      "("^(show_type false p)^" -> "^(show_type true q)^")"
  | TCon(k,a,_) -> k^"<"^(String.concat ~sep:", " (List.map a ~f:(show_type true)))^">"
                 
let string_of_type = show_type true

let makeTID context = 
  (TID(fst context), (fst context+1, snd context))

let bindTID i t (next, bindings) = 
  (next, (i,t) :: bindings)

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
  if not (is_polymorphic t) then t else 
  match t with
  | TCon(c,xs,_) -> kind c (xs |> List.map ~f:(applyContext k))
  | TID(j) ->
    match List.Assoc.find ~equal:(fun a b -> a = b) (snd k) j with
    | None -> TID(j)
    | Some(tp) -> applyContext k tp



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
  let t1 = applyContext context t1 in
  let t2 = applyContext context t2 in
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

let instantiate_type (n,m) t = 
  let substitution = ref [] in
  let next = ref n in
  let rec instantiate j =
    if not (is_polymorphic j) then j else 
    match j with
    | TID(i) -> (try TID(List.Assoc.find_exn ~equal:(fun a b -> a = b) !substitution i)
		 with Not_found -> 
                   substitution := (i,!next)::!substitution; next := (1+ !next); TID(!next-1)
		)
    | TCon(k,js,_) -> kind k (List.map ~f:instantiate js)
  in let q = instantiate t in
  (q,(!next,m))



  

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
let instantiated_type universal_type requested_type = 
  try
    let (universal_type,c) = instantiate_type empty_context universal_type in
    let (requested_type,c) = instantiate_type c requested_type in
    let c = unify c universal_type requested_type in
    Some(canonical_type (applyContext c universal_type))
  with _ -> None

let application_type f x = 
  let (f,c1) = instantiate_type empty_context f in
  let (x,c2) = instantiate_type c1 x in
  let (r,c3) = makeTID c2 in
  let c4 = unify c3 f (x @> r) in
  canonical_type (applyContext c4 r)

let argument_request request left = 
  let (request,c1) = instantiate_type empty_context request in
  let (left,c2) = instantiate_type c1 left in
  match left with
  | TID(_) -> TID(0)
  | TCon(_,[right;result],_) -> 
      let c3 = unify c2 request result in
      canonical_type (applyContext c3 right)
  | _ -> raise (Failure ("invalid function type: "^(string_of_type left)))

let function_request request = 
  canonical_type (make_arrow (TID(next_type_variable request)) request)

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
let tboolean = make_ground "boolean";;
let t0 = TID(0);;
let t1 = TID(1);;
let t2 = TID(2);;
let t3 = TID(3);;
let t4 = TID(4);;
let tlist t = kind "list" [t];;
let tstring = make_ground "string";;


(* let test_type () =  *)
(*   print_string (string_of_bool (can_unify (t1 @> t1) (t0 @> (tint @> t0)))); *)
(*   let (t0,k) = makeTID empty_context in *)
(*   let (t1,k) = makeTID k in *)
(*   let (t2,k) = makeTID k in *)
(*   let a = t1 in *)
(*   let b = tlist t2 in *)
(*   Printf.printf "%s\t%s\n" (string_of_type a) (string_of_type b); *)
(*   ignore(unify k a b); *)
(*   print_string (string_of_bool (can_unify (t1 @> t1) (t0 @> (tint @> t0)))) *)
(*   (\* print_string (string_of_type @@ t1 @> (t2 @> t2) @> tint); *)
(*   print_string (string_of_bool (can_unify (t1 @> t1) (make_arrow t1 t1))); *)
(*   print_string (string_of_bool (can_unify (make_arrow t1 t1) (make_arrow (make_arrow t1 t2) t3))); *)
(*   print_string (string_of_bool (not (can_unify (make_arrow t1 t1) (make_arrow (make_arrow t1 t2) (make_ground "int"))))); *\) *)
(* ;; *)
(* (\* test_type ();; *\) *)
