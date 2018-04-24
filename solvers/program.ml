open Core
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

let is_base_primitive = function
  |Primitive(_,_,_) -> true
  |_ -> false

let program_children = function
  | Abstraction(b) -> [b]
  | Apply(m,n) -> [m;n]
  | _ -> []

let rec application_function = function
  | Apply(f,x) -> application_function f
  | e -> e

let rec application_parse = function
  | Apply(f,x) ->
    let (f,arguments) = application_parse f in
    (f,arguments @ [x])
  | f -> (f,[])

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


let primitive_name = function | Primitive(_,n,_) -> n
                              | e -> raise (Failure ("primitive_name: "^string_of_program e^"not a primitive"))

let rec program_equal p1 p2 = match (p1,p2) with
  | (Primitive(_,n1,_),Primitive(_,n2,_)) -> n1 = n2
  | (Abstraction(a),Abstraction(b)) -> program_equal a b
  | (Invented(_,a),Invented(_,b)) -> program_equal a b
  | (Index(a),Index(b)) -> a = b
  | (Apply(a,b), Apply(x,y)) -> program_equal a x && program_equal b y
  | _ -> false

let rec compare_program p1 p2 = match (p1,p2) with
  (* Negative if p1 is smaller; 0 if they are equal; positive if p1 is bigger *)
  (* intuitively calculates (p1 - p2) *)
  | (Index(i),Index(j)) -> i - j
  | (Index(_),_) -> -1
  | (Abstraction(b1),Abstraction(b2)) -> compare_program b1 b2
  | (Abstraction(_),_) -> -1
  | (Apply(p,q),Apply(m,n)) ->
    let c = compare_program p m in
    if c = 0 then compare_program q n else c
  | (Apply(_,_),_) -> -1
  | (Primitive(_,n1,_),Primitive(_,n2,_)) -> String.compare n1 n2
  | (Primitive(_,_,_),_) -> -1
  | (Invented(_,b1),Invented(_,b2)) -> compare_program b1 b2
  | (Invented(_,_),_) -> -1
                                               

let rec infer_program_type context environment p : tContext*tp = match p with
  | Index(j) ->
    applyContext context (List.nth_exn environment j)
  | Primitive(t,_,_) -> instantiate_type context t
  | Invented(t,_) -> instantiate_type context t
  | Abstraction(b) ->
    let (xt,context) = makeTID context in
    let (context,rt) = infer_program_type context (xt::environment) b in
    applyContext context (xt @> rt)
  | Apply(f,x) ->
    let (rt,context) = makeTID context in
    let (context, xt) = infer_program_type context environment x in
    let (context, ft) = infer_program_type context environment f in
    let context = unify context ft (xt @> rt) in
    applyContext context rt

let closed_inference = snd % infer_program_type empty_context [];;


exception UnknownPrimitive of string

let every_primitive : (program String.Table.t) = String.Table.create();;


let lookup_primitive n =
  try
    Hashtbl.find_exn every_primitive n
  with _ -> raise (UnknownPrimitive n)

  
let rec evaluate (environment: 'b list) (p:program) : 'a =
  match p with
  | Apply(Apply(Apply(Primitive(_,"if",_),branch),yes),no) ->
    if magical (evaluate environment branch) then evaluate environment yes else evaluate environment no
  | Abstraction(b) -> magical @@ fun argument -> evaluate (argument::environment) b
  | Index(j) -> magical @@ List.nth_exn environment j
  | Apply(f,x) -> (magical @@ evaluate environment f) (magical @@ evaluate environment x)
  | Primitive(_,_,v) -> magical (!v)
  | Invented(_,i) -> evaluate [] i

let rec analyze_evaluation (p:program) : 'b list -> 'a =
  match p with
  | Apply(Apply(Apply(Primitive(_,"if",_),branch),yes),no) ->
    let branch = analyze_evaluation branch
    and yes = analyze_evaluation yes
    and no = analyze_evaluation no
    in
    fun environment -> 
      if magical (branch environment) then yes environment else no environment
  | Abstraction(b) ->
    let body = analyze_evaluation b in
    fun environment -> magical (fun x -> body (x::environment))
  | Index(j) ->
    fun environment -> List.nth_exn environment j |> magical
  | Apply(f,x) ->
    let analyzed_function = analyze_evaluation f
    and analyzed_argument = analyze_evaluation x
    in
    fun environment -> magical ((analyzed_function environment) (magical (analyzed_argument environment)))
  | Primitive(_,_,v) ->
    fun _ -> magical (!v)
  | Invented(_,i) ->
    let analyzed_body = analyze_evaluation i in
    fun _ -> analyzed_body []

let run_with_arguments (p : program) (arguments : 'a list) =
  let rec loop l xs =
    match xs with
    | [] -> magical l
    | x :: xs -> loop (magical (l x)) xs
  in loop (evaluate [] p) arguments

let run_analyzed_with_arguments (p : 'b list -> 'c) (arguments : 'a list) =
  let rec loop l xs =
    match xs with
    | [] -> magical l
    | x :: xs -> loop (magical (l x)) xs
  in loop (p []) arguments

let rec lazy_evaluate (environment: ('b Lazy.t) list) (p:program) : 'a Lazy.t =
  (* invariant: always return thunks *)
  match p with
  (* Notice that we do not need to special case conditionals. In lazy
     evaluation conditionals are function just like any other. *)
  | Abstraction(b) -> lazy (magical @@ fun argument -> Lazy.force (lazy_evaluate (argument::environment) b))
  | Index(j) -> magical @@ List.nth_exn environment j
  | Apply(f,x) ->
    lazy ((Lazy.force @@ magical @@ lazy_evaluate environment f) (magical @@ lazy_evaluate environment x))
  | Primitive(_,_,v) -> lazy (magical (!v))
  | Invented(_,i) -> lazy_evaluate [] i

let rec analyze_lazy_evaluation (p:program) : (('b Lazy.t) list) -> 'a Lazy.t =
  match p with
  (* Notice that we do not need to special case conditionals. In lazy
     evaluation conditionals are function just like any other. *)
  | Abstraction(b) ->
    let body = analyze_lazy_evaluation b in
    fun environment -> 
    lazy (magical @@ fun argument -> Lazy.force (body (argument::environment)))
  | Index(j) ->
    fun environment -> magical @@ List.nth_exn environment j
  | Apply(f,x) ->
    let analyzed_function = analyze_lazy_evaluation f
    and analyzed_argument = analyze_lazy_evaluation x
    in
    fun environment -> 
    lazy ((Lazy.force @@ magical @@ analyzed_function environment) (magical @@ analyzed_argument environment))
  | Primitive(_,_,v) -> fun _ -> lazy (magical (!v))
  | Invented(_,i) ->
    let analyzed_body = analyze_lazy_evaluation i in
    fun _ -> analyzed_body []

let run_lazy_analyzed_with_arguments p arguments =
  let rec go l xs =
    match xs with
    | [] -> l |> magical
    | x :: xs -> go (lazy x |> magical l) xs
  in go (p [] |> Lazy.force) arguments

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

let rec free_variables ?d:(d=0) e = match e with
  | Index(j) -> if j >= d then [j] else []
  | Apply(f,x) -> free_variables ~d:d f @ free_variables ~d:d x
  | Abstraction(b) -> free_variables ~d:(d + 1) b
  | _ -> []

(* PRIMITIVES *)
let primitive ?manualLaziness:(manualLaziness = false)
    (name : string) (t : tp) x =
  let number_of_arguments = arguments_of_type t |> List.length in
  (* Force the arguments *)
  let x = if manualLaziness then x else magical @@ 
      match number_of_arguments with
      | 0 -> magical x
      | 1 -> fun a -> (magical x) (Lazy.force a)
      | 2 -> fun a -> fun b -> (magical x) (Lazy.force a) (Lazy.force b)
      | 3 -> fun a -> fun b -> fun c -> (magical x) (Lazy.force a) (Lazy.force b) (Lazy.force c)
      | 4 -> fun a -> fun b -> fun c -> fun d -> (magical x) (Lazy.force a) (Lazy.force b) (Lazy.force c) (Lazy.force d)
      | 5 -> fun a -> fun b -> fun c -> fun d -> fun e -> (magical x) (Lazy.force a) (Lazy.force b) (Lazy.force c) (Lazy.force d) (Lazy.force e)
      | 6 -> fun a -> fun b -> fun c -> fun d -> fun e -> fun f -> (magical x) (Lazy.force a) (Lazy.force b) (Lazy.force c) (Lazy.force d) (Lazy.force e) (Lazy.force f)
      | 7 -> fun a -> fun b -> fun c -> fun d -> fun e -> fun f -> fun g -> (magical x) (Lazy.force a) (Lazy.force b) (Lazy.force c) (Lazy.force d) (Lazy.force e) (Lazy.force f) (Lazy.force g)
      | 8 -> fun a -> fun b -> fun c -> fun d -> fun e -> fun f -> fun g -> fun h -> (magical x) (Lazy.force a) (Lazy.force b) (Lazy.force c) (Lazy.force d) (Lazy.force e) (Lazy.force f) (Lazy.force g) (Lazy.force h)
      | _ ->
        raise (Failure (Printf.sprintf "Primitive %s can not be lazy because it has %d arguments. Change `primitive` in program.ml if you want to enable laziness for %s.\n"
                          name number_of_arguments name))
  in
  let p = Primitive(t,name, ref (magical x)) in
  assert (not (Hashtbl.mem every_primitive name));
  ignore(Hashtbl.add every_primitive name p);
  p

(* let primitive_empty_string = primitive "emptyString" tstring "";; *)
let primitive_uppercase = primitive "caseUpper" (tcharacter @> tcharacter) Char.uppercase;;
(* let primitive_uppercase = primitive "strip" (tstring @> tstring) (fun s -> String.strip s);; *)
let primitive_lowercase = primitive "caseLower" (tcharacter @> tcharacter) Char.lowercase;;
let primitive_character_equal = primitive "char-eq?" (tcharacter @> tcharacter @> tboolean) Char.equal;;
(* let primitive_capitalize = primitive "caseCapitalize" (tstring @> tstring) String.capitalize;;
 * let primitive_concatenate = primitive "concatenate" (tstring @> tstring @> tstring) ( ^ );; *)
let primitive_constant_strings = [primitive "','" tcharacter ',';
                                  primitive "'.'" tcharacter '.';
                                  primitive "'@'" tcharacter '@';
                                  primitive "SPACE" tcharacter ' ';
                                  primitive "'<'" tcharacter '<';
                                  primitive "'>'" tcharacter '>';
                                  primitive "'/'" tcharacter '/';
                                  primitive "'|'" tcharacter '|';
                                  primitive "'-'" tcharacter '-';
                                  primitive "LPAREN" tcharacter '(';
                                  primitive "RPAREN" tcharacter ')';
                                 ];;
(* let primitive_slice_string = primitive "slice-string" (tint @> tint @> tstring @> tstring)
 *     (fun i j s ->
 *        let i = i + (if i < 0 then String.length s else 0) in
 *        let j = j + (if j < 0 then 1 + String.length s else 0) in
 *        String.sub s ~pos:i ~len:(j - i));;
 * let primitive_nth_string = primitive "nth" (tint @> tlist tstring @> tstring)
 *     (fun n words ->
 *        let n = n + (if n < 0 then List.length words else 0) in
 *        List.nth_exn words n);;
 * let primitive_map_string = primitive "map-string" ((tstring @> tstring) @> tlist tstring @> tlist tstring)
 *     (fun f l -> List.map ~f:f l);;
 * let primitive_string_split = primitive "split" (tcharacter @> tstring @> tlist tstring)
 *     (fun d x -> String.split ~on:d x);;
 * let primitive_string_join = primitive "join" (tstring @> tlist tstring @> tstring)
 *     (fun d xs -> join ~separator:d xs);;
 * let primitive_character_to_string = primitive "chr2str" (tcharacter @> tstring) (String.of_char);; *)




let primitive0 = primitive "0" tint 0;;
let primitive1 = primitive "1" tint 1;;
let primitiven1 = primitive "-1" tint (0-1);;
let primitive2 = primitive "2" tint 2;;
let primitive3 = primitive "3" tint 3;;
let primitive4 = primitive "4" tint 4;;
let primitive5 = primitive "5" tint 5;;
let primitive6 = primitive "6" tint 6;;
let primitive7 = primitive "7" tint 7;;
let primitive8 = primitive "8" tint 8;;
let primitive9 = primitive "9" tint 9;;
let primitive_addition = primitive "+" (tint @> tint @> tint) (fun x y -> x + y);;
let primitive_increment = primitive "incr" (tint @> tint) (fun x -> 1+x);;
let primitive_decrement = primitive "decr" (tint @> tint) (fun x -> x - 1);;
let primitive_subtraction = primitive "-" (tint @> tint @> tint) (-);;
let primitive_negation = primitive "negate" (tint @> tint) (fun x -> 0-x);;
let primitive_multiplication = primitive "*" (tint @> tint @> tint) ( * );;
let primitive_modulus = primitive "mod" (tint @> tint @> tint) (fun x y -> x mod y);;

let primitive_apply = primitive "apply" (t1 @> (t1 @> t0) @> t0) (fun x f -> f x);;

let primitive_true = primitive "true" tboolean true;;
let primitive_false = primitive "false" tboolean false;;

let primitive_if = primitive "if" (tboolean @> t0 @> t0 @> t0)
    ~manualLaziness:true
    (fun p x y -> if Lazy.force p then Lazy.force x else Lazy.force y);;

let primitive_is_square = primitive "is-square" (tint @> tboolean)
    (fun x ->
       let y = Float.of_int x in
       let s = sqrt y |> Int.of_float in
       s*s = x);;
let primitive_is_prime = primitive "is-prime" (tint @> tboolean)
    (List.mem [2; 3; 5; 7; 11; 13; 17; 19; 23; 29; 31; 37; 41; 43; 47; 53; 59; 61; 67; 71; 73; 79; 83; 89; 97; 101; 103; 107; 109; 113; 127; 131; 137; 139; 149; 151; 157; 163; 167; 173; 179; 181; 191; 193; 197; 199]);;


let primitive_cons = primitive "cons" (t0 @> tlist t0 @> tlist t0) (fun x xs -> x :: xs);;
let primitive_car = primitive "car" (tlist t0 @> t0) (fun xs -> List.hd_exn xs);;
let primitive_cdr = primitive "cdr" (tlist t0 @> tlist t0) (fun xs -> List.tl_exn xs);;
let primitive_is_empty = primitive "empty?" (tlist t0 @> tboolean)
    (function | [] -> true
              | _ -> false);;

      
let primitive_empty = primitive "empty" (tlist t0) [];;
let primitive_range = primitive "range" (tint @> tlist tint) (fun x -> 0 -- (x-1));;
let primitive_sort = primitive "sort" (tlist tint @> tlist tint) (List.sort ~cmp:(fun x y -> x - y));;
let primitive_reverse = primitive "reverse" (tlist tint @> tlist tint) (List.rev);;
let primitive_append = primitive "append"  (tlist tint @> tlist tint @> tlist tint) (@);;
let primitive_singleton = primitive "singleton"  (tint @> tlist tint) (fun x -> [x]);;
let primitive_slice = primitive "slice" (tint @> tint @> tlist tint @> tlist tint) slice;;
let primitive_length = primitive "length" (tlist t0 @> tint) (List.length);;
let primitive_map = primitive "map" ((t0 @> t1) @> (tlist t0) @> (tlist t1)) (fun f l -> List.map ~f:f l);;
let primitive_fold_right = primitive "fold_right" ((tint @> tint @> tint) @> tint @> (tlist tint) @> tint) (fun f x0 l -> List.fold_right ~f:f ~init:x0 l);;
let primitive_mapi = primitive "mapi" ((tint @> t0 @> t1) @> (tlist t0) @> (tlist t1)) (fun f l ->
    List.mapi l ~f:f);;
let primitive_a2 = primitive "++" ((tlist t0) @> (tlist t0) @> (tlist t0)) (@);;
let primitive_reducei = primitive "reducei" ((tint @> t1 @> t0 @> t1) @> t1 @> (tlist t0) @> t1) (fun f x0 l -> List.foldi ~f:f ~init:x0 l);;
let primitive_filter = primitive "filter" ((tint @> tboolean) @> (tlist tint) @> (tlist tint)) (fun f l -> List.filter ~f:f l);;
let primitive_equal = primitive "eq?" (tint @> tint @> tboolean) (fun (a : int) (b : int) -> a = b);;
let primitive_equal0 = primitive "eq0" (tint @> tboolean) (fun (a : int) -> a = 0);;
let primitive_not = primitive "not" (tboolean @> tboolean) (not);;
let primitive_and = primitive "and" (tboolean @> tboolean @> tboolean) (fun x y -> x && y);;
let primitive_nand = primitive "nand" (tboolean @> tboolean @> tboolean) (fun x y -> not (x && y));;
let primitive_or = primitive "or" (tboolean @> tboolean @> tboolean) (fun x y -> x || y);;
let primitive_greater_than = primitive "gt?" (tint @> tint @> tboolean) (fun (x: int) (y: int) -> x > y);;

let primitive_run   = primitive
                        "run"
                        (tprogram @> tcanvas)
                        (fun x ->
                          GeomLib.Plumbing.relist
                            (GeomLib.Plumbing.run x))

let primitive_just     = primitive "just"
                          (t0 @> tmaybe t0)
                          (fun x -> Some(x))

let primitive_nothing= primitive "nothing" (tmaybe t0) None

let primitive_nop    = primitive "nop"  tprogram GeomLib.Plumbing.nop
let primitive_nop2   = primitive "nop2" tprogram GeomLib.Plumbing.nop
let primitive_embed  = primitive
                        "embed"
                        (tprogram @> tprogram)
                        GeomLib.Plumbing.embed
let primitive_concat = primitive
                        "concat"
                        (tprogram @> tprogram @> tprogram)
                        GeomLib.Plumbing.concat
let primitive_turn   = primitive
                        "turn"
                        (tmaybe tvar @> tprogram)
                        GeomLib.Plumbing.turn
let primitive_define = primitive
                        "define"
                        (tvar @> tprogram)
                        GeomLib.Plumbing.define
let primitive_repeat = primitive
                        "repeat"
                        (tprogram @> tprogram)
                        GeomLib.Plumbing.repeat
let primitive_integrate= primitive
                        "integrate"
                        (tmaybe tvar @> tboolean @>
                         (*tmaybe tvar @> tmaybe tvar  @>*)
                         tmaybe tvar @> tmaybe tvar  @>
                         tprogram)
                        GeomLib.Plumbing.integrate

let var_unit         = primitive "var_unit" tvar GeomLib.Plumbing.var_unit
let var_double       = primitive "var_double" (tvar @> tvar) GeomLib.Plumbing.var_double
let var_half         = primitive "var_half" (tvar @> tvar) GeomLib.Plumbing.var_half
let var_next         = primitive "var_next" (tvar @> tvar) GeomLib.Plumbing.var_next
let var_prev         = primitive "var_prev" (tvar @> tvar) GeomLib.Plumbing.var_prev
let var_opposite     = primitive "var_opposite" (tvar @> tvar) GeomLib.Plumbing.var_opposite
let var_name         = primitive "var_name" tvar GeomLib.Plumbing.var_name


let default_recursion_limit = 20;;

let rec unfold x p h n =
  if p x then [] else h x :: unfold (n x) p h n

let primitive_unfold = primitive "unfold" (t0 @> (t0 @> tboolean) @> (t0 @> t1) @> (t0 @> t0) @> tlist t1) unfold;;
let primitive_index = primitive "index" (tint @> tlist t0 @> t0) (fun j l -> List.nth_exn l j);;
let primitive_zip = primitive "zip" (tlist t0 @> tlist t1 @> (t0 @> t1 @> t2) @> tlist t2)
    (fun x y f -> List.map2_exn x y ~f:f);;
let primitive_fold = primitive "fold" (tlist t0 @> t1 @> (t0 @> t1 @> t1) @> t1)
    (fun l x0 f -> List.fold_right ~f:f ~init:x0 l);;


let default_recursion_limit = ref 50;;
let set_recursion_limit l = default_recursion_limit := l;;
exception RecursionDepthExceeded of int;;
    
let fixed_combinator argument body = 
  (* strict with respect to body but lazy with respect argument *)
  (* body expects to be passed 2 thunks *)
  let body = Lazy.force body in
  let recursion_limit = ref !default_recursion_limit in

  let rec fix x =
    (* r is just a wrapper over fix that counts the number of
       recursions *)
    let r z =
      decr recursion_limit;
      if !recursion_limit > 0 then fix z
      else raise (RecursionDepthExceeded(!default_recursion_limit))
    in
    body (lazy r) x
  in

  fix argument

let fixed_combinator2 argument1 argument2 body =
  let body = Lazy.force body in
  let recursion_limit = ref !default_recursion_limit in

  let rec fix x y = 
    let r a b =
      decr recursion_limit;
      if !recursion_limit > 0 then  
        fix a b
      else raise (RecursionDepthExceeded(!default_recursion_limit))
    in body (lazy r) x y
  in

  fix argument1 argument2 (* (lazy argument1) (lazy argument2) *)


let primitive_recursion =
  primitive ~manualLaziness:true "fix1" (t0 @> ((t0 @> t1) @> (t0 @> t1)) @> t1)
    fixed_combinator;;
let primitive_recursion2 =
  primitive ~manualLaziness:true "fix2" (t0 @> t1 @> ((t0 @> t1 @> t2) @> (t0 @> t1 @> t2)) @> t2)
    fixed_combinator2;;


let is_recursion_of_arity a = function
  | Primitive(_,n,_) -> ("fix"^(Int.to_string a)) = n
  | _ -> false

let is_recursion_primitive = function
  | Primitive(_,"fix1",_) -> true
  | Primitive(_,"fix2",_) -> true
  | _ -> false


let program_parser : program parsing =
  let token = token_parser (fun c -> Char.is_alphanum c || List.mem ~equal:( = )
                                       ['_';'-';'?';'/';'.';'*';'\'';'+';',';
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
    let rec nabstractions n b =
      if n = 0 then b else nabstractions (n-1) (Abstraction(b))
    in
    constant_parser "(lambda"%%(fun _ ->
        whitespace%%(fun _ ->
            program_parser()%%(fun b ->
                constant_parser ")"%%(fun _ ->
                    return_parse (Abstraction(b)))))
        <|>
        number%%(fun n -> whitespace%%(fun _ ->
            program_parser()%%(fun b ->
                constant_parser ")"%%(fun _ ->
                  return_parse (nabstractions (Int.of_string n) b))))))
                           
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

(* let test_program_inference program desired_type =
 *   let (context,t) = infer_program_type empty_context [] program in
 *   let t = applyContext context t in
 *   let t = canonical_type t in
 *   Printf.printf "%s : %s\n" (string_of_program program) (string_of_type t);
 *   assert (t = (canonical_type desired_type))
 * 
 * let program_test_cases() =
 *   test_program_inference (Abstraction(Index(0))) (t0 @> t0);
 *   test_program_inference (Abstraction(Abstraction(Apply(Index(0),Index(1))))) (t0 @> (t0 @> t1) @> t1);
 *   test_program_inference (Abstraction(Abstraction(Index(1)))) (t0 @> t1 @> t0);
 *   test_program_inference (Abstraction(Abstraction(Index(0)))) (t0 @> t1 @> t1);
 *   let v : int = evaluate [] (Apply(primitive_increment, primitive0)) in
 *   Printf.printf "%d\n" v;
 *   
 * ;; *)

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
  let e = parse_program "(lambda (fix (lambda (lambda (if (empty? $0) $0 (cons (* 2 (car $0)) ($1 (cdr $0)))))) $0))" |> get_some in
  let xs = [2;1;9;3;] in
  let n = 10000000 in
  time_it "evaluate program many times" (fun () -> 
      (0--n) |> List.iter ~f:(fun j ->
          if j = n then
            Printf.printf "%s\n" (evaluate [] e xs |> List.map ~f:Int.to_string |> join ~separator:" ")
          else 
            ignore(evaluate [] e xs)));
  let c = analyze_evaluation e [] in
  time_it "evaluate analyzed program many times" (fun () -> 
      (0--n) |> List.iter ~f:(fun j ->
          if j = n then
            Printf.printf "%s\n" (c xs |> List.map ~f:Int.to_string |> join ~separator:" ")
          else 
            ignore(c xs)))
;;


(* performance_test_case();; *)


(* let recursion_test_case() =
 *   let f zs = fixed_combinator zs (fun r l ->
 *       match l with
 *       | [] -> []
 *       | x::xs -> x*2 :: r xs) in
 *   f (0--18) |> List.map ~f:Int.to_string |> join ~separator:" " |> Printf.printf "%s\n";
 *   f (0--10) |> List.map ~f:Int.to_string |> join ~separator:" " |> Printf.printf "%s\n";
 *   f (0--2) |> List.map ~f:Int.to_string |> join ~separator:" " |> Printf.printf "%s\n";
 *   let e = parse_program "(lambda (fix1 (lambda (lambda (if (empty? $0) $0 (cons (\* 2 (car $0)) ($1 (cdr $0)))))) $0))" |> get_some in
 *   Printf.printf "%s\n" (string_of_program e);
 *   evaluate [] e [1;2;3;4;] |> List.map ~f:Int.to_string |> join ~separator:" " |> Printf.printf "%s\n";
 * 
 *   let e = parse_program "(lambda (lambda (fix2 (lambda (lambda (lambda (if (empty? $1) $0 (cons (car $1) ($2 (cdr $1) $0)))))) $0 $1)))" |> get_some in
 *   infer_program_type empty_context [] e |> snd |> string_of_type |> Printf.printf "%s\n";
 *   evaluate [] e (0--4) [9;42;1] |> List.map ~f:Int.to_string |> join ~separator:" " |> Printf.printf "%s\n" *)

(* recursion_test_case();; *)

(* let timeout_test_cases() = *)
(*   let list_of_numbers = [ *)
(*     "(lambda (fix (lambda (lambda (if (empty? $0) $0 (cons (\* 2 (car $0)) ($1 (cdr $0)))))) $0))"; *)

(*   ] in *)

(*   let list_of_numbers = list_of_numbers |> List.map ~f:(analyze_evaluation%get_some%parse_program) in *)

(*   let xs = [(0--10);(0--10);(0--10)] in *)

(*   time_it "evaluated all of the programs" (fun () -> *)
      
  

(* let () = *)
(*   let e = parse_program "(lambda (reducei (lambda (lambda (lambda (range $0)))) empty $0))" |> get_some in *)
(*   Printf.printf "tp = %s\n" (string_of_type @@ snd @@ infer_program_type empty_context [] e); *)
(*   let f = evaluate [] e in *)
(*   f [1;2]; *)
(*   List.foldi [1;3;2;]  ~init:[]  ~f:(fun x y z -> 0--z) |> List.iter ~f:(fun a -> *)
(*       Printf.printf "%d\n" a) *)

(* let () = *)
(*   let e = parse_program "(lambda (lambda (- $1 $0)))" |> get_some in *)
(*   Printf.printf "%d\n" (run_with_arguments e [1;9]) *)
let test_lazy_evaluation() =
  let ps = ["1";"0";"(+ 1 1)";
            "(lambda (+ $0 2))"; "(+ 5)";
            "-"; "(lambda2 (- $0 $1))";
            "((lambda 1) (car empty))";
            "((lambda $0) 9)";
            "((lambda ($0 ($0 ($0 1)))) (lambda (+ $0 $0)))";
            "((lambda (lambda (if (eq? $0 0) $1 (+ $1 $1)))) 5 1)";
            "((lambda2 (if (eq? $0 0) $1 (+ $1 $1))) 5 0)";
            "(car (cdr (cons 1 (cons 2 (cons 3 empty)))))";
            "(cdr (cons 1 (cons 2 (cons 3 empty))))";
            "(map (+ 1) (cons 1 (cons 2 (cons 3 empty))))";
            "(map +1 (cons 1 (cons 2 (cons 3 empty))))";
            "(map (lambda (+ $0 $0)) (cons 1 (cons 2 (cons 3 empty))))";
            "(fold_right (lambda2 (+ $0 $1)) 0 (cons 1 (cons 2 (cons 3 empty))))";
            "(fix1 (cons 1 (cons 2 (cons 3 empty))) (lambda2 (if (empty? $0) 0 (+ 1 ($1 (cdr $0))))))";
            "(fix1 (cons 1 (cons 2 (cons 3 empty))) (lambda2 (if (empty? $0) 0 (+ (car $0) ($1 (cdr $0))))))";
           "(fix2 5 7 (lambda (lambda (lambda (if (eq? $0 0) 0 (+ $1 ($2 $1 (- $0 1))))))))"] in
  ps |> List.iter ~f:(fun p ->
      Printf.printf "About to evaluate the following program lazily: %s\n" p;
      flush_everything();
      let p = parse_program p |> get_some in
      let t = infer_program_type empty_context [] p |> snd in
      let a = analyze_lazy_evaluation p in
      let arguments = match List.length (arguments_of_type t) with
        | 0 -> []
        | 1 -> [42]
        | 2 -> [0;1]
      in
      Printf.printf "\t(arguments: %s)\n"
        (arguments |> List.map ~f:Int.to_string |> join ~separator:"; ");
      flush_everything();
      let v = run_lazy_analyzed_with_arguments a arguments in
      begin 
        match string_of_type (return_of_type t) with
        | "int" -> 
          Printf.printf "value = %d\n" (v |> magical)
        | "list<int>" ->
          Printf.printf "value = %s\n" (v |> magical |> List.map ~f:Int.to_string |> join ~separator:",")
      end
      ;
      flush_everything()
    );;

let test_string() =
  let p = parse_program "(lambda (fold $0 $0 (lambda (lambda (cdr (if (char-eq? $1 SPACE) $2 $0))))))" |> get_some in
  let p = analyze_lazy_evaluation p in
  let x = String.to_list "this is a rigorous" in
  let y = run_lazy_analyzed_with_arguments p [x] |> String.of_char_list in
  Printf.printf "%s\n" y
;;

