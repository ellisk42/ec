open Core

type str = String of char list | Dot | D | S | W | L | U

type pregex = 
	| Constant of str
	| Kleene of pregex
	| Plus of pregex
	| Maybe of pregex
	| Alt of pregex * pregex
	| Concat of pregex * pregex;;

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
		| Dot -> if in_list dot_ls char_list then let _::t = char_list in [ (None, t), 0. ] else []
		| D -> if in_list d_ls char_list then let _::t = char_list in [ (None, t), 0. ] else []
		| S -> if in_list s_ls char_list then let _::t = char_list in [ (None, t), 0. ] else []
		| W -> if in_list w_ls char_list then let _::t = char_list in [ (None, t), 0. ] else []
		| L -> if in_list l_ls char_list then let _::t = char_list in [ (None, t), 0. ] else []
		| U -> if in_list u_ls char_list then let _::t = char_list in [ (None, t), 0. ] else []
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
	((None, str), log 0.5 ) :: match consume (Some(a), str) with 
								| [] -> [] (*maybe not needed?*)
								| ls -> List.map ls ~f:(f_kleene a)

and plusConsume a str = 
	match consume (Some(a), str) with 
	| [] -> []
	| ls -> List.map ls ~f:(f_plus a)

and maybeConsume a str = 
	((None, str), log 0.5 ) :: match consume (Some(a), str) with
								| [] -> [] (*maybe not needed?*)
								| ls -> List.map ls ~f:(f_maybe a)

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

	let cmp = fun (_, score1) (_, score2) -> if score2 -. score1 > 0. then 1 else -1 in 
	let heap = Heap.create ~cmp:cmp () in
	Heap.add heap ((Some(preg), str), 0.);
	let visited = Hash_set.Poly.create() in
	let solution = ref None in

	let consume_loop (cont_old, score_old) = 
		List.map consume cont ~f:(fun (cont, score) ->
			if not Hash_set.mem visited cont then
				Hash_set.add visited cont;
				let newscore = score +. score_old in
					match cont with
					| (None, []) -> 
						solution := newscore; () (* TODO output *)
					| _ -> 
						Heap.add heap (cont, newscore); () ) (* TODO output *) in

	while !solution = None && not Heap.top heap = None do
		let partial = Heap.pop heap in 
		consume_loop partial;
	done;

	match !solution with
	| None -> log 0.
	| score -> score ;;




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
