open Core

open Timeout
open Task
open Utils
open Program
open Type

    
type str = String of char list | Dot | D | S | W | L | U
[@@deriving compare]           

type pregex = 
	| Constant of str
	| Kleene of pregex
	| Plus of pregex
	| Maybe of pregex
	| Alt of pregex * pregex
	| Concat of pregex * pregex
[@@deriving compare]

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
                   

let rec canonical_regex r = match r with
  | Constant(_) -> r
  | Plus(b) ->
    let b = canonical_regex b in 
    canonical_regex (Concat(b,Kleene(b)))
  | Kleene(b) -> Kleene(canonical_regex b)
  | Maybe(b) -> Maybe(canonical_regex b)
  (* associative rules *)
  | Concat(Concat(a,b),c) -> canonical_regex (Concat(a,Concat(b,c)))
  | Alt(Alt(a,b),c) -> canonical_regex (Alt(a,Alt(b,c)))
  | Concat(a,b) -> Concat(canonical_regex a, canonical_regex b)
  | Alt(a,b) ->
    let a = canonical_regex a in
    let b = canonical_regex b in
    match compare_pregex a b with
    | 0 -> a
    | d when d > 0 -> Alt(a,b)
    | d when d < 0 -> Alt(b,a)
    | _ -> assert false 

type continuation = pregex option * string


let dot_ls = List.rev (String.to_list_rev "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t")
let d_ls = List.rev (String.to_list_rev "0123456789")
let s_ls = List.rev (String.to_list_rev " \t")
let w_ls = List.rev (String.to_list_rev "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
let l_ls = List.rev (String.to_list_rev "abcdefghijklmnopqrstuvwxyz")
let u_ls = List.rev (String.to_list_rev "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

let rec try_remove_prefix prefix str = 
	(* returns a none if can't remove, else the removed list*)
	match (prefix, str) with
	| ([], l) -> Some(l)
	| (_, []) ->  None
	| (ph::pt, sh::st) -> if ph = sh then try_remove_prefix pt st else None;;

let rec in_list ls str =
	(* checks if there's an element in ls which can be the first element in str*) 
	match (ls, str) with
	| ([], l) -> false
	| (_, []) -> false
	| (h::t, x::xs) -> if h = x then true else in_list t (x::xs);;

let consumeConst c char_list = 
	match char_list with 
	| [] -> []
	| char_list -> 
		match c with 
		| Dot -> if in_list dot_ls char_list then let _::t = char_list in [ (None, t), -. log (List.length dot_ls |> Float.of_int) ] else []
		| D -> if in_list d_ls char_list then let _::t = char_list in [ (None, t), -. log (List.length d_ls |> Float.of_int) ] else []
		| S -> if in_list s_ls char_list then let _::t = char_list in [ (None, t), -. log (List.length s_ls |> Float.of_int) ] else []
		| W -> if in_list w_ls char_list then let _::t = char_list in [ (None, t), -. log (List.length w_ls |> Float.of_int) ] else []
		| L -> if in_list l_ls char_list then let _::t = char_list in [ (None, t), -. log (List.length l_ls |> Float.of_int) ] else []
		| U -> if in_list u_ls char_list then let _::t = char_list in [ (None, t), -. log (List.length u_ls |> Float.of_int) ] else []
		| String(ls) -> 
			match try_remove_prefix ls char_list with 
				| None -> []
				| Some(remainder) -> [ (None, remainder), 0. ] ;;

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
		| (None, []) -> [] (* needed here?*)
		| (Some(Constant(c)), str) -> consumeConst c str
		| (Some(Kleene(a)), str) -> kleeneConsume a str 
		| (Some(Plus(a)), str) -> plusConsume a str 
		| (Some(Maybe(a)), str) -> maybeConsume a str
		| (Some(Alt(a,b)), str) -> List.map (List.concat [(consume (Some(a), str)); (consume (Some(b), str))]) ~f:f_alt
		| (Some(Concat(a,b)), str) -> concatConsume a b str ;;



let preg_match preg str = (* dikjstras *)

	let cmp = fun (_, score1) (_, score2) -> Float.compare score2 score1 in 
	let heap = Heap.create ~cmp:cmp () in
	Heap.add heap ((Some(preg), str), 0.);
	let visited = Hash_set.Poly.create() in
	let solution = ref None in

	let consume_loop (cont_old, score_old) = 
		consume cont_old |> List.iter ~f:(fun (cont, score) ->
			if not (Hash_set.mem visited cont) then
				Hash_set.add visited cont;
				let newscore = score +. score_old in
					match cont with
					| (None, []) -> 
						solution := Some(newscore) (* TODO output *)
					| _ -> 
						Heap.add heap (cont, newscore) ) (* TODO output *) in

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
ignore(primitive "r_kleene" ((tregex @> tregex) @> tregex @> tregex)
         (fun b k -> Concat(Kleene(b empty_regex),k)));;
ignore(primitive "r_plus" ((tregex @> tregex) @> tregex @> tregex)
         (fun b k -> Concat(Plus(b empty_regex),k)));;
ignore(primitive "r_maybe" ((tregex @> tregex) @> tregex @> tregex)
         (fun b k -> Concat(Maybe(b empty_regex),k)));;
ignore(primitive "r_alt" ((tregex @> tregex) @> (tregex @> tregex) @> tregex @> tregex)
         (fun a b k -> Concat(Alt(a empty_regex, b empty_regex),k)));;


let regex_of_program expression : pregex =
  run_lazy_analyzed_with_arguments (analyze_lazy_evaluation expression) [empty_regex];;


register_special_task "regex"
  (fun extra ?timeout:(timeout=0.001)
    name task_type examples ->
    assert (task_type = tregex @> tregex);
    examples |> List.iter ~f:(fun (xs,_) -> assert (List.length xs = 0));

    let observations : char list list = examples |> List.map ~f:(fun (_,y) -> y |> magical) in 

    let log_likelihood expression =
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
      | Some(l) -> l
      | None -> log 0.
    in 

    {name; task_type; log_likelihood;} )
                                

(* let _ = Printf.printf "%f\n"  (preg_match (Alt(Constant(String(['d'])), Constant(D)))  ['9'] |> exp)  *)


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
