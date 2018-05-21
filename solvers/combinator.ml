open Core.Std

open Program
open Type

(* type cl = *)
(*   | Branch of cl*cl *)
(*   | Leaf of tp*string *)

let cS = Primitive((t0 @> t1 @> t2) @> ((t0 @> t1) @> (t0 @> t2)), "S");;
let cI = Primitive(t0 @> t0, "I");;
let cB = Primitive((t1 @> t2) @> ((t0 @> t1) @> (t0 @> t2)), "B");;
let cC = Primitive((t0 @> t1 @> t2) @> t1 @> t0 @> t2, "C");;
let cK = Primitive(t0 @> (t1 @> t0), "K");;


let rec program_to_combinator = function
  | Invented(t,p) -> Invented(t,p)
  | Primitive(t,n) -> Primitive(t,n)
  | Index(j) -> Index(j)
  | Apply(f,x) -> Apply(program_to_combinator f,program_to_combinator x)
  | Abstraction(Index(0)) -> cI
  (* for when we don't use the variable in the body *)
  | Abstraction(b) when not (variable_is_bound b) -> Apply(cK,program_to_combinator b)
  | Abstraction(Abstraction(b)) ->
    let b = program_to_combinator (Abstraction(b)) |> shift_free_variables (-1) in
    program_to_combinator (Abstraction(b))
  | Abstraction(Apply(f,x)) when not (variable_is_bound f) ->
    (* variable must be bound in x *)
    let f = program_to_combinator f in
    Apply(Apply(cB, f), program_to_combinator (Abstraction(x)))
  | Abstraction(Apply(f,x)) when not (variable_is_bound x) ->
    (* variable must be bound in x *)
    let x = program_to_combinator x in
    Apply(Apply(cC, program_to_combinator (Abstraction(f))), x)
  | Abstraction(Apply(f,x)) ->
    (* variable must occur free in both function and argument *)
    Apply(Apply(cS, program_to_combinator (Abstraction(f))),
          program_to_combinator (Abstraction(x)))
  | _ -> raise (Failure "program to combinator")

let combinator_to_program p = 
  let rec combinator_to_program_ = function    
    | Primitive(_,"I") -> Abstraction(Index(0))
    | Primitive(_,"K") -> Abstraction(Abstraction(Index(1)))
    | Primitive(_,"S") -> Abstraction(Abstraction(Abstraction(Apply(Apply(Index(2),Index(0)),
                                                                    Apply(Index(1),Index(0))))))
    | Primitive(_,"C") -> Abstraction(Abstraction(Abstraction(Apply(Apply(Index(2),Index(0)),
                                                                    Index(1)))))
    | Primitive(_,"B") -> Abstraction(Abstraction(Abstraction(Apply(Index(2),
                                                                    Apply(Index(1),Index(0))))))
    | Invented(t,p) -> Invented(t,p)
    | Apply(f,x) -> Apply(combinator_to_program_ f, combinator_to_program_ x)
    | Abstraction(_) -> raise (Failure "abstraction in combinator")
    | Index(_) -> raise (Failure "index in combinator")
    | Primitive(t,n) -> Primitive(t,n)
  in
  let rec reduce p = match p with
    | Abstraction(b) -> begin match reduce b with
        | None -> None
        | Some(b) -> Some(Abstraction(b))
      end
    | Apply(Abstraction(b),x) -> begin match reduce x with
        | Some(x) -> Some(Apply(Abstraction(b),x))
        | None -> Some(substitute x b)
      end
    | Apply(f,x) -> begin match reduce f with
        | Some(f) -> Some(Apply(f,x))
        | None -> match reduce x with
          | None -> None
          | Some(x) -> Some(Apply(f,x))
      end
    | _ -> None
  and substitute ?height:(height = 0) value body = match body with
    | Apply(f,x) -> Apply(substitute ~height:height value f,
                          substitute ~height:height value x)
    | Abstraction(b) -> Abstraction(substitute ~height:(height+1) value b)
    | Index(j) when j = height -> shift_free_variables height value
    | _ -> body
  in
  let rec repeatedly_reduce p = match reduce p with
    | None -> p
    | Some(p) -> repeatedly_reduce p
  in
  combinator_to_program_ p |> repeatedly_reduce
  

let test_combinator() =
  [cS;cI;cB;cC;cK;] |> List.iter ~f:(fun (Primitive(t,n)) ->
      Printf.printf "%s : %s\n" n (string_of_type t));
  List.iter [Abstraction(Abstraction(Index(1)));
             Abstraction(Abstraction(Index(0)));
             Abstraction(Apply(Apply(primitive_multiplication,Index(0)),Index(0)));
             Abstraction(Abstraction(Apply(Apply(primitive_multiplication,Index(0)),Index(1))));
             Abstraction(Abstraction(Abstraction(Apply(Apply(primitive_multiplication,Index(0)),Index(2)))));
             Abstraction(Abstraction(Abstraction(Apply(Apply(primitive_multiplication,Index(0)),Index(1)))));]
    ~f:(fun p ->
      Printf.printf "\n%s : %s\n" (string_of_program p) (infer_program_type empty_context [] p |> snd |> string_of_type);
      let p = program_to_combinator p in
      Printf.printf "%s :" (string_of_program p);
      Out_channel.flush stdout;
      Printf.printf " %s\n" (infer_program_type empty_context [] p |> snd |> string_of_type);
      let p = combinator_to_program p in
      Printf.printf "%s :" (string_of_program p);
      Out_channel.flush stdout;
      Printf.printf " %s\n" (infer_program_type empty_context [] p |> snd |> string_of_type))
;;


test_combinator();;
