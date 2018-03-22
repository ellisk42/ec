open Core

open Utils
open Type

type fast_type =
  | FastVariable of tp option ref
  | FastConstructor of string * (fast_type list) * (tp option)

let rec fast_return = function
  | FastConstructor("->", [_;r], _) -> fast_return r
  | t -> t

let rec fast_arguments = function
  | FastConstructor("->", [a;r], _) -> a :: fast_arguments r
  | _ -> []

let fast_polymorphic = function
  | FastConstructor(_,_,Some(_)) -> true
  | FastConstructor(_,_,None) -> false
  | FastVariable(r) -> match !r with
    | None -> true
    | Some(t) -> is_polymorphic t

let make_fast t =
  let t = canonical_type t in
  let nt = next_type_variable t in
  let mapping = Array.init nt ~f:(fun _ -> ref None) in

  let rec make = function
    | TID(j) -> FastVariable(Array.get mapping j)
    | TCon(n,xs,p) as t ->
      FastConstructor(n, xs |> List.map ~f:make, if p then None else Some(t))
  in
  (make t, mapping)

let rec make_slow next_type_variable = function
  | FastConstructor(n,xs,_) ->
    kind n (xs |> List.map ~f:(make_slow next_type_variable))
  | FastVariable(r) -> match !r with
    | Some(t) -> t
    | None ->
      let t = TID(!next_type_variable) in
      incr next_type_variable;
      r := Some(t);
      t

let compile_unifier t =
  let (f, mapping) = make_fast t in
  let return_type = fast_return f in
  let argument_types = fast_arguments f in

  (* todo because f is known ahead of time we can compile this into closures *)
  let rec fu f t k : tContext = match (f,t) with
    | (FastVariable(r), t) -> begin 
      match !r with
        | None -> r := Some(t); k
        | Some(t') -> unify k t t'
    end
    | (FastConstructor(n,fs,_),TCon(n',ss,_)) ->
      if n = n' then
        List.fold2_exn ~init:k ~f:(fun k f s -> fu f s k) fs ss
      else raise UnificationFailure
    | (FastConstructor(_,_,Some(t')),TID(j)) ->
      bindTID j t' k
    | (FastConstructor(_,[],None),_) -> assert false (* should be impossible *)
    | (FastConstructor(n,[f1],None),TID(j)) ->
      let (t',k) = makeTID k in
      let k = bindTID j (TCon(n,[t'],true)) k in
      fu f1 t' k
    | (FastConstructor(n,[f1;f2],None),TID(j)) ->
      let (t1,k) = makeTID k in
      let (t2,k) = makeTID k in
      let k = bindTID j (TCon(n,[t1;t2;],true)) k in
      let k = fu f1 t1 k in
      let k = fu f2 t2 k in
      k
    | (FastConstructor(n,[f1;f2;f3],None),TID(j)) ->
      let (t1,k) = makeTID k in
      let (t2,k) = makeTID k in
      let (t3,k) = makeTID k in
      let k = bindTID j (TCon(n,[t1;t2;t3;],true)) k in
      let k = fu f1 t1 k in
      let k = fu f2 t2 k in
      let k = fu f3 t3 k in
      k

    | _ -> raise (Failure "Fast types do not currently support arity >=3 type constructors")

  in

  fun context request ->
    let context = fu return_type request context in
    let next_type_variable = fst context in
    let next_type_variable' = ref next_type_variable in
    let arguments = List.map argument_types ~f:(make_slow next_type_variable') in
    let context = makeTIDs (!next_type_variable' - next_type_variable) context in

    Array.iter mapping ~f:(fun r -> r := None);

    (context, arguments)
     

let test_fast() =
  let library_types = [t1 @> t0 @> tlist t0 @> tlist t0] in
  let requesting_types = [tlist tint; tlist t1; t2] in

  library_types |> List.iter ~f:(fun library_type -> 
      let u = compile_unifier library_type in
      requesting_types |> List.iter ~f:(fun request -> 
          let k = makeTIDs (next_type_variable request) empty_context in

          let (k,arguments) = u k request in
          Printf.printf "library type: %s\trequesting type: %s\t\n"
            (string_of_type library_type) (string_of_type request);
          Printf.printf "arguments:\t%s\n"
            (arguments |> List.map ~f:(snd%applyContext k) |> List.map ~f:string_of_type |> join ~separator:"\t");
          Printf.printf "unified request:\t%s\n"
            ([request] |> List.map ~f:(snd%applyContext k) |> List.map ~f:string_of_type |> join ~separator:"\t");
          Printf.printf "\n"
        ))

;;

(* test_fast();; *)
