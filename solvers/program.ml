open Core.Std

open Parser
open Utils
open Type

type program =
  | Index of int
  | Abstraction of program
  | Apply of program*program
  | Primitive of tp * string * (unit ref)
  | Invented of tp * program

let is_index = function
  |Index(_) -> true
  |_ -> false

let is_primitive = function
  |Primitive(_,_,_) -> true
  |Invented(_,_) -> true
  |_ -> false

let program_children = function
  | Abstraction(b) -> [b]
  | Apply(m,n) -> [m;n]
  | _ -> []

let rec program_size p =
  1 + (List.map ~f:program_size (program_children p) |> sum)

let rec program_subexpressions p =
  p::(List.map (program_children p) program_subexpressions |> List.concat)

let rec show_program (is_function : bool) = function
  | Index(j) -> "$" ^ string_of_int j
  | Abstraction(body) ->
    "(lambda "^show_program false body^")"
  | Apply(p,q) ->
    if is_function then
      show_program true p^" "^show_program false q
    else
      "("^show_program true p^" "^show_program false q^")"
  | Primitive(_,n,_) -> n
  | Invented(_,i) -> "#"^show_program false i

let string_of_program = show_program false

let rec infer_program_type context environment = function
  | Index(j) ->
    let (t,context) = List.nth_exn environment j |> chaseType context in (context,t)
  | Primitive(t,_,_) -> let (t,context) = instantiate_type context t in (context,t)
  | Invented(t,_) -> let (t,context) = instantiate_type context t in (context,t)
  | Abstraction(b) ->
    let (xt,context) = makeTID context in
    let (context,rt) = infer_program_type context (xt::environment) b in
    let (ft,context) = chaseType context (xt @> rt) in
    (context,ft)
  | Apply(f,x) ->
    let (rt,context) = makeTID context in
    let (context, xt) = infer_program_type context environment x in
    let (context, ft) = infer_program_type context environment f in
    let context = unify context ft (xt @> rt) in
    let (rt, context) = chaseType context rt in
    (context, rt)

exception UnknownPrimitive of string

let every_primitive : (program String.Table.t) = String.Table.create();;


let lookup_primitive n =
  try
    Hashtbl.find_exn every_primitive n
  with _ -> raise (UnknownPrimitive n)

  
let rec evaluate (environment: 'b list) (p:program) : 'a =
  match p with
  | Abstraction(b) -> magical @@ fun argument -> evaluate (argument::environment) b
  | Index(j) -> magical @@ List.nth_exn environment j
  | Apply(f,x) -> (magical @@ evaluate environment f) (magical @@ evaluate environment x)
  | Primitive(_,_,v) -> magical (!v)
  | Invented(_,i) -> evaluate [] i

let rec remove_abstractions (n : int) (q : program) : program =
  match (n,q) with
  | (0,q) -> q
  | (n,Abstraction(body)) -> remove_abstractions (n - 1) body
  | _ -> raise (Failure "remove_abstractions")

let rec variable_is_bound ?height:(height = 0) (p : program) =
  match p with
  | Index(j) -> j = height
  | Apply(f,x) -> variable_is_bound ~height:height f || variable_is_bound ~height:height x
  | Invented(_,i) -> variable_is_bound ~height:height i
  | Primitive(_,_,_) -> false
  | Abstraction(b) -> variable_is_bound ~height:(height+1) b

let rec shift_free_variables ?height:(height = 0) shift p = match p with
  | Index(j) -> if j < height then p else Index(j + shift)
  | Apply(f,x) -> Apply(shift_free_variables ~height:height shift f,
                        shift_free_variables ~height:height shift x)
  | Invented(_,_) -> p
  | Primitive(_,_,_) -> p
  | Abstraction(b) -> Abstraction(shift_free_variables ~height:(height+1) shift b)

(* PRIMITIVES *)
let primitive (name : string) (t : tp) x =
  let p = Primitive(t,name, ref (magical x)) in
  assert (not (Hashtbl.mem every_primitive name));
  ignore(Hashtbl.add every_primitive name p);
  p

let primitive_empty_string = primitive "emptyString" tstring "";;
let primitive_uppercase = primitive "caseUpper" (tstring @> tstring) String.uppercase;;
let primitive_uppercase = primitive "strip" (tstring @> tstring) String.strip;;
let primitive_lowercase = primitive "caseLower" (tstring @> tstring) String.lowercase;;
let primitive_capitalize = primitive "caseCapitalize" (tstring @> tstring) String.capitalize;;
let primitive_concatenate = primitive "concatenate" (tstring @> tstring @> tstring) ( ^ );;
let primitive_constant_strings = [primitive "','" tcharacter ',';
                                  primitive "'.'" tcharacter '.';
                                  primitive "'@'" tcharacter '@';
                                  primitive "SPACE" tcharacter ' ';
                                  primitive "'<'" tcharacter '<';
                                  primitive "'>'" tcharacter '>';
                                  primitive "'/'" tcharacter '/';
                                  primitive "'|'" tcharacter '|';
                                  primitive "'-'" tcharacter '-';
                                 ];;
let primitive_slice_string = primitive "slice-string" (tint @> tint @> tstring @> tstring)
    (fun i j s ->
       let i = i + (if i < 0 then String.length s else 0) in
       let j = j + (if j < 0 then 1 + String.length s else 0) in
       String.sub s ~pos:i ~len:(j - i));;
let primitive_nth_string = primitive "nth" (tint @> tlist tstring @> tstring)
    (fun n words ->
       let n = n + (if n < 0 then List.length words else 0) in
       List.nth_exn words n);;
let primitive_map_string = primitive "map-string" ((tstring @> tstring) @> tlist tstring @> tlist tstring)
    (fun f l -> List.map ~f:f l);;
let primitive_string_split = primitive "split" (tcharacter @> tstring @> tlist tstring)
    (fun d x -> String.split ~on:d x);;
let primitive_string_join = primitive "join" (tstring @> tlist tstring @> tstring)
    (fun d xs -> join ~separator:d xs);;
let primitive_character_to_string = primitive "chr2str" (tcharacter @> tstring) (String.of_char);;




let primitive0 = primitive "0" tint 0;;
let primitive1 = primitive "1" tint 1;;
let primitive2 = primitive "2" tint 2;;
let primitive3 = primitive "3" tint 3;;
let primitive4 = primitive "4" tint 4;;
let primitive5 = primitive "5" tint 5;;
let primitive6 = primitive "6" tint 6;;
let primitive7 = primitive "7" tint 7;;
let primitive8 = primitive "8" tint 8;;
let primitive9 = primitive "9" tint 9;;
let primitive_addition = primitive "+" (tint @> tint @> tint) (+);;
let primitive_increment = primitive "+1" (tint @> tint) (fun x -> 1+x);;
let primitive_decrement = primitive "-1" (tint @> tint) (fun x -> x - 1);;
let primitive_subtraction = primitive "-" (tint @> tint @> tint) (-);;
let primitive_negation = primitive "negate" (tint @> tint) (fun x -> 0-x);;
let primitive_multiplication = primitive "*" (tint @> tint @> tint) ( * );;
let primitive_modulus = primitive "mod" (tint @> tint @> tint) (fun x y -> x mod y);;

let primitive_apply = primitive "apply" (t1 @> (t1 @> t0) @> t0) (fun x f -> f x);;

let primitive_true = primitive "true" tboolean true;;
let primitive_false = primitive "false" tboolean false;;

let primitive_if = primitive "if" (tboolean @> t0 @> t0 @> t0)
    (fun p x y -> if p then x else y);;

let primitive_is_square = primitive "is-square" (tint @> tboolean)
    (fun x ->
       let y = Float.of_int x in
       let s = sqrt y |> Int.of_float in
       s*s = x);;
let primitive_is_prime = primitive "is-prime" (tint @> tboolean)
    (List.mem [2; 3; 5; 7; 11; 13; 17; 19; 23; 29; 31; 37; 41; 43; 47; 53; 59; 61; 67; 71; 73; 79; 83; 89; 97; 101; 103; 107; 109; 113; 127; 131; 137; 139; 149; 151; 157; 163; 167; 173; 179; 181; 191; 193; 197; 199]);;


let primitive_cons = primitive "cons" (t0 @> tlist t0 @> tlist t0) (fun x xs -> x :: xs);;

      
let primitive_empty = primitive "empty" (tlist t0) [];;
let primitive_range = primitive "range" (tint @> tlist tint) (fun x -> 0 -- (x-1));;
let primitive_sort = primitive "sort" (tlist tint @> tlist tint) (List.sort ~cmp:(fun x y -> x - y));;
let primitive_reverse = primitive "reverse" (tlist tint @> tlist tint) (List.rev);;
let primitive_append = primitive "append"  (tlist tint @> tlist tint @> tlist tint) (@);;
let primitive_singleton = primitive "singleton"  (tint @> tlist tint) (fun x -> [x]);;
let primitive_slice = primitive "slice" (tint @> tint @> tlist tint @> tlist tint) slice;;
let primitive_length = primitive "length" (tlist tint @> tint) (List.length);;
let primitive_map = primitive "map" ((tint @> tint) @> (tlist tint) @> (tlist tint)) (fun f l -> List.map ~f:f l);;
let primitive_fold_right = primitive "fold_right" ((tint @> tint @> tint) @> tint @> (tlist tint) @> tint) (fun f x0 l -> List.fold_right ~f:f ~init:x0 l);;
let primitive_mapi = primitive "mapi" ((tint @> t0 @> t1) @> (tlist t0) @> (tlist t1)) (fun f l ->
    List.mapi l ~f:f);;
let primitive_a2 = primitive "++" ((tlist t0) @> (tlist t0) @> (tlist t0)) (@);;
let primitive_reducei = primitive "reducei" ((tint @> t1 @> t0 @> t1) @> t1 @> (tlist t0) @> t1) (fun f x0 l -> List.foldi ~f:f ~init:x0 l);;
let primitive_filter = primitive "filter" ((tint @> tboolean) @> (tlist tint) @> (tlist tint)) (fun f l -> List.filter ~f:f l);;
let primitive_equal = primitive "eq?" (tint @> tint @> tboolean) ( = );;
let primitive_not = primitive "not" (tboolean @> tboolean) (not);;
let primitive_and = primitive "and" (tboolean @> tboolean @> tboolean) (fun x y -> x && y);;
let primitive_or = primitive "or" (tboolean @> tboolean @> tboolean) (fun x y -> x || y);;
let primitive_greater_than = primitive "gt?" (tint @> tint @> tboolean) (fun (x: int) (y: int) -> x > y);;


let program_parser : program parsing = 
  let token = token_parser (fun c -> Char.is_alphanum c || List.mem ['_';'-';'?';'/';'.';'*';'\'';'+';',';
                                                                    '>';'<';'@';'|';] c) in
  let whitespace = token_parser ~can_be_empty:true Char.is_whitespace in
  let number = token_parser Char.is_digit in
  let primitive = token %% (fun name ->
      try
        return_parse (lookup_primitive name)
      with _ ->
        (*         Printf.printf "Error finding type of primitive %s\n" name; *)
        parse_failure)
  in
  let variable : program parsing = constant_parser "$" %% (fun _ ->
      number%%(fun n -> Index(Int.of_string n) |> return_parse))
  in
      
  let rec program_parser () : program parsing =
    (application () <|> primitive <|> variable <|> invented() <|> abstraction())

  and invented() =
    constant_parser "#" %% (fun _ ->
        program_parser()%%(fun p ->
            let t =
              try
                infer_program_type empty_context [] p |> snd
              with UnificationFailure -> begin
                  Printf.printf "WARNING: Could not type check invented %s\n" (string_of_program p);
                  t0
                end
            in
            return_parse (Invented(t,p))))

  and abstraction() =
    constant_parser "(lambda"%%(fun _ ->
        whitespace%%(fun _ ->
            program_parser()%%(fun b ->
                constant_parser ")"%%(fun _ ->
                    return_parse (Abstraction(b))))))
                           
  and application_sequence (maybe_function : program option) : program parsing =
    whitespace%%(fun _ ->
        match maybe_function with
        | None -> (* cannot terminate sequence because there is nothing before it *)
          program_parser () %%(fun f -> application_sequence (Some(f)))
        | Some(f) ->
          (return_parse f) <|> (program_parser () %%(fun x -> application_sequence (Some(Apply(f,x))))))
        
    
  and application () =
    constant_parser "(" %% (fun _ ->
        application_sequence None %% (fun a -> 
            constant_parser ")" %% (fun _ ->
                return_parse a)))
  in

  program_parser ()

let parse_program s = run_parser program_parser s

let test_program_inference program desired_type =
  let (context,t) = infer_program_type empty_context [] program in
  let (t,_) = chaseType context t in
  let t = canonical_type t in
  Printf.printf "%s : %s\n" (string_of_program program) (string_of_type t);
  assert (t = (canonical_type desired_type))

let program_test_cases() =
  test_program_inference (Abstraction(Index(0))) (t0 @> t0);
  test_program_inference (Abstraction(Abstraction(Apply(Index(0),Index(1))))) (t0 @> (t0 @> t1) @> t1);
  test_program_inference (Abstraction(Abstraction(Index(1)))) (t0 @> t1 @> t0);
  test_program_inference (Abstraction(Abstraction(Index(0)))) (t0 @> t1 @> t1);
  let v : int = evaluate [] (Apply(primitive_increment, primitive0)) in
  Printf.printf "%d\n" v;
  
;;

let parsing_test_case s =
  Printf.printf "Parsing the string %s\n" s;
  program_parser s |> List.iter ~f:(fun (p,suffix) ->
      if suffix = "" then
        (Printf.printf "Parsed into the program: %s\n" (string_of_program p);
         assert (s = (string_of_program p)))
      else
        Printf.printf "With the suffix %s, we get the program %s\n" suffix (string_of_program p));
  Printf.printf "\n"
;;

let parsing_test_cases() =
  parsing_test_case "+1";
  parsing_test_case "($0 $1)";
  parsing_test_case "(+1 $0 $2)";
  parsing_test_case "(map +1 $0 $1)";
  parsing_test_case "(map +1 ($0 +1 -1 (+ -)) $1)";
  parsing_test_case "(lambda $0)";
  parsing_test_case "(lambda (+ k1 #(* k8 k1)))";
  parsing_test_case "(lambda (+ k1 #(* k8 map)))";
;;


(* parsing_test_cases();; *)


(* program_test_cases();; *)
             
let performance_test_case() =
  let e = parse_program "((lambda (+ $0 $0)) k1)" |> get_some in
  time_it "evaluate program many times" (fun () -> 
      (0--5000000) |> List.iter ~f:(fun j ->
          if j = 5000000 then
            Printf.printf "%d\n" (evaluate [] e)
          else 
            ignore(evaluate [] e)));;


(* performance_test_case();; *)


(* let () = *)
(*   let e = parse_program "(lambda (reducei (lambda (lambda (lambda (range $0)))) empty $0))" |> get_some in *)
(*   Printf.printf "tp = %s\n" (string_of_type @@ snd @@ infer_program_type empty_context [] e); *)
(*   let f = evaluate [] e in *)
(*   f [1;2]; *)
(*   List.foldi [1;3;2;]  ~init:[]  ~f:(fun x y z -> 0--z) |> List.iter ~f:(fun a -> *)
(*       Printf.printf "%d\n" a) *)
