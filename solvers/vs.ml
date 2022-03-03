open Core

open Program
open Utils
open Type

type vs = 
    Single of program
  | ApplySpace of vs*vs
  | AbstractSpace of vs
  | UnionSpace of vs list
  | Universe
  | Vacuum

let rec shift_space ?height:(height = 0) n = function
  | Single(e) -> Single(shift_free_variables ~height:height n e)
  | ApplySpace(f,x) -> ApplySpace(shift_space ~height:height n f,shift_space ~height:height n x)
  | AbstractSpace(b) -> AbstractSpace(shift_space ~height:(height+1) n b)
  | UnionSpace(u) -> UnionSpace(u |> List.map ~f:(shift_space ~height:height n))
  | Universe -> Universe
  | Vacuum -> Vacuum

let rec compare_space u v = match (u,v) with
  | (Single(x),Single(y)) -> compare_program x y
  | (Single(_),_) -> 1
  | (ApplySpace(a,b),ApplySpace(x,y)) ->
    let c = compare_space a x in
    if c = 0 then compare_space b y else c
  | (ApplySpace(_,_),_) -> 1
  | (AbstractSpace(a),AbstractSpace(b)) -> compare_space a b
  | (AbstractSpace(_),_) -> 1
  | (UnionSpace(a),UnionSpace(b)) ->
    compare_list compare_space a b
  | (UnionSpace(_),_) -> 1
  | (Universe,Universe) -> 0
  | (Universe,_) -> 1
  | (Vacuum,Vacuum) -> 0
  | (Vacuum,_) -> 1

let rec space_elements = function
  | Single(e) -> [e]
  | ApplySpace(f,x) ->
    let f = space_elements f in
    let x = space_elements x in
    f |> List.map ~f:(fun f' -> x |> List.map ~f:(fun x' -> Apply(f',x'))) |> List.concat
  | AbstractSpace(b) ->
    space_elements b |> List.map ~f:(fun b -> Abstraction(b))
  | Vacuum -> []
  | Universe -> [Primitive(t0,"??",ref ())]
  | UnionSpace(u) -> u |> List.map ~f:space_elements |> List.concat 
                

let union ss =
  let ss = ss |> List.filter ~f:(function | Vacuum -> false | _ -> true) |> 
           List.dedup_and_sort ~compare:compare_space
  in 
  if List.exists ss ~f:(function | Universe -> true | _ -> false)
  then Universe else
    match ss with
    | [] -> Vacuum
    | [v] -> v
    | _ ->
      let not_a_union = ss |> List.filter ~f:(function| UnionSpace(_) -> false | _ -> true) in
      let extra = ss |> List.filter_map ~f:(function| UnionSpace(x) -> Some(x) | _ -> None)
                  |> List.concat in
      UnionSpace(not_a_union @ extra)

let rec inverse_beta (s : vs) : vs =
  let inverse_e s = ApplySpace(AbstractSpace(shift_space 1 s), Universe) in

  (* let rec subtrees (d : int) : vs -> vs list = function
   *   | Vacuum | Universe -> []
   *   | *) 
    
  
  match s with
  | Vacuum | Universe -> s
  | UnionSpace(u) ->
    union (inverse_e s :: u |> List.map ~f:inverse_beta)
  | AbstractSpace(b) ->
    union [AbstractSpace(inverse_beta b); inverse_e s]
  | ApplySpace(f,x) ->
    union [ApplySpace(inverse_beta f, x);
           ApplySpace(f, inverse_beta x);
           inverse_e s]
  | Single(e) ->
    union [inverse_e s;
           match e with
           | Apply(f,x) -> union [ApplySpace(inverse_beta (Single(f)), Single(x));
                                  ApplySpace(Single(f), inverse_beta (Single(x)))]
           | Abstraction(b) -> AbstractSpace(inverse_beta (Single(b)))
           | _ -> Vacuum]


let rec intersect_single p v = match (p,v) with
  | (_,Single(p')) ->
    compare_program p p' = 0
  | (Apply(f,x),ApplySpace(f',x')) -> intersect_single f f' || intersect_single x x'
  | (Abstraction(body),AbstractSpace(body')) -> intersect_single body body'
  | (_,UnionSpace(u)) -> u |> List.exists ~f:(intersect_single p)
  | (_,Universe) -> true
  | (_,Vacuum) -> false
  | _ -> false

let rec intersect a b = match (a,b) with
  | (Vacuum,_) | (_,Vacuum) -> Vacuum
  | (Universe,x) | (x,Universe) -> x
  | (ApplySpace(f,x),ApplySpace(f',x')) ->
    ApplySpace(intersect f f', intersect x x')
  | (AbstractSpace(u),AbstractSpace(v)) -> AbstractSpace(intersect u v)
  | (ApplySpace(_,_), AbstractSpace(_)) | (AbstractSpace(_), ApplySpace(_,_)) -> Vacuum
  | (UnionSpace(u),z) | (z,UnionSpace(u)) ->
    union (u |> List.map ~f:(intersect z))
  | (Single(p),v) | (v,Single(p)) ->
    if intersect_single p v then Single(p) else Vacuum

(* let rec closed_subtrees ?d:(d=0) e =
 *   let this = try shift_free_variables
 *     with ShiftFailure -> []
 *   in
 *   match e with
 *   | Index(j) ->
 *     if j <= d then [Index(j - d)] else []
 *   | Apply(f,x) -> closed_subtrees ~d:d f @ closed_subtrees ~d:d x @
 *                   
 * 
 * let test_version() =
 *   let e = "(lambda (+ 1 (car $0)))" |> parse_program |> get_some in
 *   let v = inverse_beta (Single(e)) in
 *   space_elements v |> List.iter ~f:(fun e ->
 *       Printf.printf "%s\n" (string_of_program e))
 * ;; *)

  
