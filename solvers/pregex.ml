open Core

open Timeout
open Task
open Utils
open Program
open Type
open Yojson.Basic.Util

    
type str = String of char list | Dot | D | S | W | L | U
[@@deriving compare]

let dot_ls = List.rev (String.to_list_rev "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t")
let d_ls = List.rev (String.to_list_rev "0123456789")
let s_ls = List.rev (String.to_list_rev " \t")
let w_ls = List.rev (String.to_list_rev "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
let l_ls = List.rev (String.to_list_rev "abcdefghijklmnopqrstuvwxyz")
let u_ls = List.rev (String.to_list_rev "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

let get_character_class = function
  | Dot -> dot_ls
  | D -> d_ls | S -> s_ls | W -> w_ls | L -> l_ls | U -> u_ls
  | _ -> assert (false)

type pregex = 
	| Constant of str
	| Kleene of pregex
	| Plus of pregex
	| Maybe of pregex
	| Alt of pregex * pregex
	| Concat of pregex * pregex
[@@deriving compare]

let rec string_of_str = function
  | String(c) -> String.of_char_list c
  | Dot -> "."
  | D -> "[0-9]"
  | S -> "\\w" | W -> "[0-9a-zA-Z]" | L -> "[a-z]" | U -> "A-Z"

let rec show_regex = function
  | Kleene(b) -> show_regex b |> Printf.sprintf "(%s)*"
  | Plus(b) -> show_regex b |> Printf.sprintf "(%s)+"
  | Maybe(b) -> show_regex b |> Printf.sprintf "(%s)?"
  | Alt(a,b) -> Printf.sprintf "(%s|%s)" (show_regex a) (show_regex b)
  | Concat(a,b) -> Printf.sprintf "%s%s" (show_regex a) (show_regex b)
  | Constant(k) -> string_of_str k

let rec hash_regex = function
  | Plus(r) -> Hashtbl.hash (hash_regex r, 0)
  | Kleene(r) -> Hashtbl.hash (hash_regex r, 1)
  | Maybe(r) -> Hashtbl.hash (hash_regex r, 2)
  | Alt(a,b) -> Hashtbl.hash (hash_regex a, hash_regex b, 3)
  | Concat(a,b) -> Hashtbl.hash (hash_regex a, hash_regex b, 4)
  | Constant(k) -> Hashtbl.hash (hash_constant k, 5)
and
  hash_constant = function
  | String(cs) -> List.fold_right ~init:17 ~f:(fun c a -> Hashtbl.hash (Hashtbl.hash c,a)) cs
  | c -> Hashtbl.hash c
                   

let rec canonical_regex r =
  let empty_regex = Constant(String([])) in
  
  match r with
  | Constant(_) -> r
  | Plus(b) ->
    let b = canonical_regex b in 
    canonical_regex (Concat(b,Kleene(b)))
  | Kleene(b) ->
    let b = canonical_regex b in
    if b = empty_regex then empty_regex else Kleene(b)    
  | Maybe(b) -> Alt(empty_regex,b) |> canonical_regex
  (* associative rules *)
  | Concat(Concat(a,b),c) -> canonical_regex (Concat(a,Concat(b,c)))
  | Alt(Alt(a,b),c) -> canonical_regex (Alt(a,Alt(b,c)))
  | Concat(a,b) ->
    let a = canonical_regex a in
    let b = canonical_regex b in
    if a = empty_regex then b else
    if b = empty_regex then a else
      Concat(a, b)
  | Alt(a,b) ->
    let a = canonical_regex a in
    let b = canonical_regex b in
    match compare_pregex a b with
    | 0 -> a
    | d when d > 0 -> Alt(a,b)
    | d when d < 0 -> Alt(b,a)
    | _ -> assert false


type match_state = {
  match_target: char list;
  match_likelihood : float;
}

let rec match_regex random (state : match_state) r return = match r with
  | Constant(String(s)) ->
    if List.take state.match_target (List.length s) = s then
      return {state with match_target=List.drop state.match_target (List.length s)}
  | Constant(k) ->
    (match state.match_target with
     | hd :: tl when List.mem ~equal:(=) (get_character_class k) hd ->
       return {match_likelihood=state.match_likelihood-.(get_character_class k |> List.length
                                                        |> Float.of_int |> log);
               match_target=tl}
     | _ -> ())

  | Concat(x,y) ->
    match_regex random state x (fun state' ->
        match_regex random state' y return)
  | Alt(x,y) -> begin
      let state' = {state with match_likelihood = state.match_likelihood-.log 2.} in
      random state'
        (fun () -> match_regex random state' x return);
      random state'
        (fun () -> match_regex random state' y return)
    end
  | Maybe(x) -> begin
      let state' = {state with match_likelihood = state.match_likelihood-.log 2.} in
      random state'
        (fun () -> match_regex random state' x return);
      random state'
        (fun () -> return state')
    end
  | Kleene(x) -> begin
      let state' = {state with match_likelihood = state.match_likelihood-.log 2.} in
      random state'
        (fun () -> match_regex random state' (Concat(x,r)) return);
      random state'
        (fun () -> return state')
    end
  | Plus(x) -> match_regex random state (Concat(x,Kleene(x))) return


let match_regex regex s =
  let h = Heap.create ~cmp:(fun (s1,_) (s2,_) -> Float.compare s2 s1) () in

  let final_state = ref None in

  match_regex (fun state k ->
      Heap.add h (state.match_likelihood, k))
    {match_target=s; match_likelihood=0.;}
    regex
    (fun final ->
       if final.match_target = [] then  
         final_state := Some(final));
  while !final_state = None && not (Heap.is_empty h) do
    let _,k = Heap.pop_exn h in
    k()
  done;
  match !final_state with
  | Some(s) -> s.match_likelihood
  | _ -> log 0.
;;

if false then begin 
  Printf.eprintf "%f\n"
    (match_regex (Concat(Kleene(Alt(Constant(D),Constant(S))),Constant(String([])))) ['9';' ';]);
  flush_everything();
  assert (false)
end;;

           
  


type continuation = pregex option * string



let rec try_remove_prefix prefix str = 
	(* returns a none if can't remove, else the removed list *)
	match (prefix, str) with
	| ([], l) -> Some(l)
	| (_, []) ->  None
	| (ph::pt, sh::st) -> if ph = sh then try_remove_prefix pt st else None;;

let consumeConst c char_list =
  match char_list with 
  | [] -> (match c with
      (* We are done! We didn't need to match anything and we have nothing *)
      | String([]) -> [(None, []), 0.]
      (* We failed - we needed to terminate matching because the string is up *)
      | _ -> [])
  | hd :: tl -> 
    match c with 
    | String(ls) -> 
      (match try_remove_prefix ls char_list with 
       | None -> []
       | Some(remainder) -> [ (None, remainder), 0. ])
    | _ ->
      let character_class = get_character_class c in 
      if List.mem ~equal:(=) character_class hd
      then [(None, tl), -. log (List.length character_class |> Float.of_int)]
      else []
        

let f_kleene a partial = 
	match partial with
	|  ((None, remainder), score) -> ((Some(Kleene(a)), remainder), score +. log 0.5)
	|  ((Some(r), remainder), score) -> ((Some(Concat(r, Kleene(a))), remainder), score +. log 0.5);; (* is this order of r and a correct?*)

let f_plus a partial = 
	match partial with
	|  ((None, remainder), score) -> ((Some(Kleene(a)), remainder), score)
	|  ((Some(r), remainder), score) -> ((Some(Concat(r, Kleene(a))), remainder), score) ;; (* is this order of r and a correct?*)

let f_maybe a ((r, remainder), score) = ((r, remainder), score +. log 0.5) ;;

let f_alt ((r, remainder), score) = ((r, remainder), score +. log 0.5 ) ;;

let f_concat a b partial = 
	match partial with 
	| ((None, remainder), score) -> ((Some(b), remainder), score)
	| ((Some(r), remainder), score) -> ((Some(Concat(r, b)), remainder), score) ;;

let rec kleeneConsume a str = 
	((None, str), log 0.5 ) :: (consume (Some(a), str) |> List.map ~f:(f_kleene a))

and plusConsume a str = 
  consume (Some(a), str) |> List.map ~f:(f_plus a)

and maybeConsume a str = 
  ((None, str), log 0.5 ) :: (consume (Some(a), str) |> List.map ~f:(f_maybe a))

and concatConsume a b str = List.map (consume (Some(a), str)) ~f:(f_concat a b)

and consume cont = (* return a list of continuation, score tuples? *)
  match cont with
  | (None, []) -> assert false (* no regular expression and nothing to consume - this is a match, and should have been caught earlier *)
  | (None, _ :: _) -> [] (* no regular expression and things to consume - not a match *)
  | (Some(Constant(c)), str) -> consumeConst c str
  | (Some(Kleene(a)), str) -> kleeneConsume a str 
  | (Some(Plus(a)), str) -> plusConsume a str 
  | (Some(Maybe(a)), str) -> maybeConsume a str
  | (Some(Alt(a,b)), str) -> List.map (consume (Some(a), str) @ consume (Some(b), str)) ~f:f_alt
  | (Some(Concat(a,b)), str) -> concatConsume a b str


let preg_match preg str = (* dikjstras *)

  let cmp = fun (_, score1) (_, score2) -> Float.compare score2 score1 in 
  let heap = Heap.create ~cmp:cmp () in
  Heap.add heap ((Some(preg), str), 0.);
  let visited = Hash_set.Poly.create() in
  let solution = ref None in

  let consume_loop (cont_old, score_old) = 
    consume cont_old |> List.iter ~f:(fun (cont, score) ->
	if not (Hash_set.mem visited cont) then begin 
	  Hash_set.add visited cont;
	  let newscore = score +. score_old in
	  match cont with
	  | (None, []) -> 
	    solution := Some(newscore) (* TODO output *)
          | _ ->
            Heap.add heap (cont, newscore)
        end)
  in 

  while !solution = None && not (Heap.top heap = None) do
    match Heap.pop heap with
    | Some(partial) -> consume_loop partial
    | None -> assert false		
  done;

  match !solution with
  | None -> log 0.
  | Some(score) -> score ;;


let tregex = make_ground "pregex" ;;
let empty_regex = Constant(String([]));;

ignore(primitive "r_dot" (tregex @> tregex)
         (fun k -> Concat(Constant(Dot),k)));;
ignore(primitive "r_d" (tregex @> tregex)
         (fun k -> Concat(Constant(D),k)));;
ignore(primitive "r_s" (tregex @> tregex)
         (fun k -> Concat(Constant(S),k)));;
ignore(primitive "r_w" (tregex @> tregex)
         (fun k -> Concat(Constant(W),k)));;
ignore(primitive "r_l" (tregex @> tregex)
         (fun k -> Concat(Constant(L),k)));;
ignore(primitive "r_u" (tregex @> tregex)
         (fun k -> Concat(Constant(U),k)));;


ignore(primitive "r_kleene" ((tregex @> tregex) @> tregex @> tregex)
         (fun b k -> Concat(Kleene(b empty_regex),k)));;
ignore(primitive "r_plus" ((tregex @> tregex) @> tregex @> tregex)
         (fun b k -> Concat(Plus(b empty_regex),k)));;
ignore(primitive "r_maybe" ((tregex @> tregex) @> tregex @> tregex)
         (fun b k -> Concat(Maybe(b empty_regex),k)));;
ignore(primitive "r_alt" ((tregex @> tregex) @> (tregex @> tregex) @> tregex @> tregex)
         (fun a b k -> Concat(Alt(a empty_regex, b empty_regex),k)));;

ignore(primitive "r_const" (tregex @> tregex)
         ());;

let rec substitute_constant_regex constant p = match p with
  | Abstraction(b) -> Abstraction(substitute_constant_regex constant b)
  | Apply(f, x) -> Apply(substitute_constant_regex constant f,
                         substitute_constant_regex constant x)
  | Invented(t,b) -> Invented(t, substitute_constant_regex constant b)
  | Primitive(t,"r_const",_) -> constant
  | Index(_) | Primitive(_,_,_) -> p

let disallowed_regex = Hashtbl.Poly.create ();;
let constant_to_regex = Hashtbl.Poly.create();;
let build_constant_regex characters =
  let rec loop = function
    | [] -> Index(0)
    | x :: xs -> Apply(Hashtbl.find_exn constant_to_regex x,
                       loop xs)
  in Abstraction(loop characters);;



[
    ('#', "hash");
    ('!', "bang");
    ('\"', "double_quote");
    ('$', "dollar");
    ('%', "percent");
    ('&', "ampersand");
    ('\'', "single_quote");
    (')', "left_paren");
    ('(', "right_paren");
    ('*', "astrisk");
    ('+', "plus");
    (',', "comma");
    ('-', "dash");
    ('.', "period");
    ('/', "slash");
    (':', "colon");
    (';', "semicolon");
    ('<', "less_than");
    ('=', "equal");
    ('>', "greater_than");
    ('?', "question_mark");
    ('@', "at");
    ('[', "left_bracket");
    ('\\', "backslash");
    (']', "right_bracket");
    ('^', "carrot");
    ('_', "underscore");
    ('`', "backtick");
    ('|', "bar");
    ('}', "right_brace");
    ('{', "left_brace");
    ('~', "tilde");
    (' ', "space");
    ('\t', "tab")
] |> List.iter ~f: (fun (c, name) -> Hashtbl.set disallowed_regex ~key:c ~data:name);;



dot_ls |> List.iter ~f: (fun i ->
    let name =
      match Hashtbl.find disallowed_regex i with
	| None -> Printf.sprintf "string_%c" i
	| Some(datum) -> Printf.sprintf "string_%s" datum
    in
    let the_primitive = primitive name (tregex @> tregex)
        (fun k -> Concat(Constant(String([i])),k)) in 
    Hashtbl.set constant_to_regex ~key:i ~data:the_primitive);;


let regex_of_program expression : pregex =
  run_lazy_analyzed_with_arguments (analyze_lazy_evaluation expression) [empty_regex];;

(* Printf.eprintf "hello world %" *)
 
register_special_task "regex"
  (fun extra ?timeout:(timeout=0.001)
    name task_type examples ->
    assert (task_type = tregex @> tregex);
    examples |> List.iter ~f:(fun (xs,_) -> assert (List.length xs = 0));

    let observations : char list list = examples |> List.map ~f:(fun (_,y) -> y |> magical) in

    let const =
      try
        let k = extra |> member "str_const" |> to_string |> String.to_list in
        if List.length k > 0 then Some(k) else None
      with _ -> None
    in
    let const_cost = match const with
      | None -> 0.
      | Some(k) -> (List.length k |> Float.of_int) *. (log 96.)
    in
    let const = match const with
      | None -> None
      | Some(k) -> Some(build_constant_regex k)
    in
    let log_likelihood expression =
      let number_of_constants = number_of_free_parameters expression in
      if number_of_constants > 1 || number_of_constants = 1 && is_none const then log 0. else
        let expression = match const with
          | Some(const) -> substitute_constant_regex const expression
          | None -> expression
        in

        match run_for_interval timeout
                (fun () ->
                   let r : pregex = regex_of_program expression in
                   let rec loop = function
                     | [] -> 0.
                     | e :: es ->
                       let this_score = preg_match r e in
                       if is_invalid this_score then log 0. else this_score +. loop es
                   in
                   loop observations)
        with
        | None -> log 0.
        | Some(l) -> 
      	  (let cutoff_option = extra |> member "cutoff" |> to_number_option in
      	   match cutoff_option with
           | None -> l        
      	   | Some(cutoff) -> 
      	     if l >= cutoff then l -. (Float.of_int number_of_constants) *. const_cost
             else log 0.)
    in 

    {name; task_type; log_likelihood;} )
                                

(* let _ = Printf.eprintf "\n\n\n\nLIKELIHOOD: %f\n\n\n\n"  (preg_match (Alt(Constant(String(['d'])), Constant(D)))  ['9'] |> exp) ;;
let _ = Printf.eprintf "\n\n\n\nLIKELIHOOD: %f\n\n\n\n"  ((preg_match (Constant(D)) ['9']) |> exp) ;; *)

(*let _ = assert (preg_match (Alt(Constant(String(['d'])), Constant(D)))  ['9'] = log (1. /. 20.) ) ;;
let _ = assert (preg_match (Alt(Constant(String(['s'])), Constant(D)))  ['9'] = log (1. /. 20.) ) ;;
let _ = assert (preg_match (Alt(Constant(String(['k'])), Constant(D)))  ['9'] = log (1. /. 20.) ) ;;*)

(* qs for kevin:
map -> List.map list ~f:fun
implicit def - good 
concat_lists -> List.concat [list1; list2 ...]
and recursion?
function defs with pattern matching?
queue
check states
Hasttbl.Poly.create()
Hash_set.Poly.create()
String.to_list()
ref and !
Hash_set.mem candidates *)

