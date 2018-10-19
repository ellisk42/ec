(* #use "topfind" *)
(* #thread *)
(* #require "core" *)
(* #require "yojson" ;; *)

open Yojson;;
open Core;;

type dir_type = 
{
	mutable delta_x : int;
	mutable delta_y : int;
}

let up = {delta_x = -1; delta_y = 0};;
let down = {delta_x = 1; delta_y = 0};;
let left = {delta_x = 0; delta_y = -1};;
let right = {delta_x = 0; delta_y = 1};;

type cell_occupancy = Blocked | Empty | Marker | Hero | Hero_and_Marker;;

type hero_type =
{
	mutable x : int;
	mutable y : int;
	mutable dir : dir_type;
};;

type game_type = 
{
	mutable hero : hero_type; 
	n : int;
	m : int;
	mutable board : (cell_occupancy array) array;
};;

let make_init_board local_n local_m hero = 
	let the_board = (Array.make_matrix local_n local_m Empty) in
	the_board.(hero.x).(hero.y) <- Hero;
	the_board;;

exception Exception of string
let dir_type_to_string x = 
	match x with
	|{delta_x = -1; delta_y = 0} -> "^"
	|{delta_x = 1; delta_y = 0} -> "V"
	|{delta_x = 0; delta_y = -1} -> "<"
	|{delta_x = 0; delta_y = 1} -> ">"
	|{delta_x = x; delta_y = y} -> raise (Exception "Direction not handled");;

exception Exception of string
let mixed_type_to_string x = 
	match x with
	|{delta_x = -1; delta_y = 0} -> "A";
	|{delta_x = 1; delta_y = 0} -> "U";
	|{delta_x = 0; delta_y = -1} -> "C";
	|{delta_x = 0; delta_y = 1} -> "D";
	|{delta_x = x; delta_y = y} -> raise (Exception "Direction not handled");;

let cell_type_to_string x hero_dir = 
	match x with
	|Blocked -> "#"
	|Empty -> "."
	|Marker -> "o"
	|Hero_and_Marker -> mixed_type_to_string hero_dir
	|Hero -> dir_type_to_string hero_dir;;

let print_row my_array hero_dir= 
	print_string "[|";
	for i = 0 to ((Array.length my_array)-1) do
		printf "%s" (cell_type_to_string my_array.(i) hero_dir)
	done;
	print_string "|]";;

let print_matrix the_matrix hero_dir = 
	print_string "[|\n";
	for i = 0 to ((Array.length the_matrix)-1) do
		if not (phys_equal i 0) then print_string "\n" else ();
		print_row the_matrix.(i) hero_dir;
	done;
	print_string "|]\n";;

let print_game game = 
	print_matrix game.board game.hero.dir;
	print_string "\n";;

let make_new_game local_n local_m = 
	let local_hero = {x = 0; y = 0; dir = right} in 
	{hero = local_hero; n = local_n; m = local_m; board = (make_init_board local_n local_m local_hero)};;

let rec set value game = function
	|[] -> ()
	|(x, y)::t -> game.board.(x).(y) <- value; set value game t;;

let remove_hero game =
	let cell = game.board.(game.hero.x).(game.hero.y) in 
	if cell = Hero_and_Marker then game.board.(game.hero.x).(game.hero.y) <- Marker else game.board.(game.hero.x).(game.hero.y) <- Empty;;

let set_hero game = 
	let cell = game.board.(game.hero.x).(game.hero.y) in 
	if cell = Marker then game.board.(game.hero.x).(game.hero.y) <- Hero_and_Marker else (game.board.(game.hero.x).(game.hero.y) <- Hero);;

let invariant game = 
	if (game.board.(game.hero.x).(game.hero.y) = Hero) || (game.board.(game.hero.x).(game.hero.y) = Hero_and_Marker) then true else false;;

let move_forward game =
	assert (invariant game);
	remove_hero game; 
	game.hero.x <- max (min (game.hero.x + game.hero.dir.delta_x) (game.n-1)) 0;
	game.hero.y <- max (min (game.hero.y + game.hero.dir.delta_y) (game.m-1)) 0;
	set_hero game;;

let put_marker game = 
	assert (invariant game);
	let hero = game.hero in
	game.board.(hero.x).(hero.y) <- Hero_and_Marker;;


let pick_marker game = 
	assert (invariant game);
	let board = game.board in
	let hero = game.hero in
	if board.(hero.x).(hero.y) = Hero_and_Marker then game.board.(hero.x).(hero.y) <- Hero
	 else ();;

let turn_left game = 
	assert (invariant game);
	let rec rotate_left = function
		|{delta_x = -1; delta_y = 0} -> {delta_x = 0; delta_y = -1}
		|{delta_x = 1; delta_y = 0} -> {delta_x = 0; delta_y = 1}
		|{delta_x = 0; delta_y = -1} -> {delta_x = 1; delta_y = 0}
		|{delta_x = 0; delta_y = 1} -> {delta_x = -1; delta_y = 0}
		|{delta_x = x; delta_y = y} -> raise (Exception "Direction not handled")
	in 
	game.hero.dir <- (rotate_left game.hero.dir);;

let turn_right game = 
	assert (invariant game);
	let rec rotate_right = function
		|{delta_x = -1; delta_y = 0} -> {delta_x = 0; delta_y = 1}
		|{delta_x = 1; delta_y = 0} -> {delta_x = 0; delta_y = -1}
		|{delta_x = 0; delta_y = -1} -> {delta_x = -1; delta_y = 0}
		|{delta_x = 0; delta_y = 1} -> {delta_x = 1; delta_y = 0}
		|{delta_x = x; delta_y = y} -> raise (Exception "Direction not handled")
	in 
	game.hero.dir <- (rotate_right game.hero.dir);;


let execute_primitives game primitives = 
	let execute_primitive primitive = 
		match primitive with
		|"turnRight" -> turn_right game
		|"move" -> move_forward game
		|"turnLeft" -> turn_left game
		|"putMarker" -> put_marker game
		|"pickMarker" -> pick_marker game
		|_ -> raise (Exception "Primitive not handled")
	in 
	let rec aux = function
		|[] -> ()
		|x :: t -> execute_primitive x; print_string x; print_string "\n"; print_game game; aux t;
	in
	aux primitives;;

let new_game = make_new_game 5 6 in

print_game new_game;

set Marker new_game [(0, 1); (1, 1); (2, 2); (3, 3)];
print_game new_game;

set Blocked new_game [(1,2); (2, 3); (3, 4)];
print_game new_game;


execute_primitives new_game 
	["move"; "pickMarker"; "turnRight"; "move"; "pickMarker"; "move"; "turnLeft"; "move"; "pickMarker"; "turnRight"; "move"; "pickMarker"; "turnLeft"; "move"; "pickMarker"; ];

(*
type if_type =
{
	if_parent : program_block_type;
	if_conditional: program_block_type;
	if_body: program_block_type list;
}
and ifelse_type =
{
	ifelse_parent: program_block_type;
	ifelse_conditional: program_block_type;
	ifsele_body: program_block_type list;
}
and else_type =
{
	else_parent: program_block_type;
	else_body: program_block_type list;
}
and main_type =
{
	main_body: program_block_type list;
}
and repeat_type =
{
	repeat_parent : program_block_type;
	repeat_count : int;
	repeat_body: program_block_type list;
}
and while_type =
{
	while_parent : program_block_type;
	while_conditional: program_block_type;
	while_body: program_block_type list;
}
and primitive_type =
{
	primitive_parent: program_block_type;
	primitive: string;
}
and program_block_type = MAIN_type of main_type| PRIMITIVE_type of primitive_type | IF_type of if_type| IFELSE_type of ifelse_type| ELSE_type of else_type| REPEAT_type of repeat_type| WHILE_type of while_type;;

let execute_program game program_instructions = 
	let rec execute_instruction open_brackets acc_body = function
		|"DEF"::t -> assert (acc_body = [] && open_brackets = []); execute_instruction open_brackets acc_body t;
		|"run"::t -> assert (acc_body = [] && open_brackets = []); execute_instruction open_brackets acc_body t;
		|"m("::t -> assert (acc_body = [] && open_brackets = []); [(MAIN_type {main_body = execute_instruction ("m("::open_brackets) [] t})] (*main*)
		|"c("::t -> execute_instruction (("c(", acc_body)::open_brackets) [] t
		(*|"w("::t -> ()(*while*)*)
		(*|"r("::t -> ()(*REPEAT*)*)
		|"i("::t -> (execute_instruction (("i(", acc_body)::open_brackets) []) (*IF/IFELSE*)
		(*|"e("::t -> ()(*ELSE*)*)
		|"m)"::t -> match open_brackets with |(start, old_acc_body)::l -> assert(start = "m(" && l = [] && t = []); (List.rev (acc_body::old_acc_body)) (*main*)
		|"c)"::t -> match open_brackets with |(start, old_acc_body)::l -> assert(start = "c("); execute_instruction l (List.rev acc_body::old_acc_body) t (*conditional*)
		(*|"w)"::t -> ()(*while*)*)
		(*|"r)"::t -> ()(*REPEAT*)*)
		|"i)"::t -> match open_brackets with |(start, old_acc_body)::l -> assert(start = "i("); execute_instruction l (List.rev acc_body::old_acc_body) t(*IF/IFELSE*)
		(*|"e)"::t -> ()(*ELSE*)*)
		(*|"markersPresent"::t -> ()*)
		|"IF"::t -> execute_instruction open_brackets acc_body t; (*(IF_type {if_conditional = execute_instruction open_brackets [] t; if_body = execute_instruction [] })*)
		(*|"IFELSE"::t -> ()*)
		(*|"ELSE"::t ->()*)
		(*|"REPEAT"::t -> ()*)
		(*|"WHILE"::t -> ()*)
		(*|"frontIsClear"::t ->()
		(*|"rightIsClear"::t ->()*)
		(*|"leftIsClear"::t -> ()*)
		|"turnRight"::t ->()
		|"move"::t -> ()
		|"turnLeft"::t -> ()
		|"putMarker"::t ->()
		|"pickMarker"::t ->()*)
		(*|r_string::t -> ()*)
	in 
	();;
*)


(*
move_forward new_game;
print_game new_game;

move_forward new_game;
print_game new_game;

put_marker new_game;
print_game new_game;

pick_marker new_game;
print_game new_game;

pick_marker new_game;
print_game new_game;

turn_left new_game;
print_game new_game;

move_forward new_game;
print_game new_game;

turn_left new_game;
print_game new_game;

move_forward new_game;
print_game new_game;

pick_marker new_game;
print_game new_game;

turn_left new_game;
print_game new_game;

move_forward new_game;
print_game new_game;

pick_marker new_game;
print_game new_game;*)

;;



(*
["DEF", "run", "m(", "REPEAT", "R=3", "r(", "IF", "c(", "not", "c(", "leftIsClear", "c)", "c)", "i(", "move", "WHILE", "c(", "noMarkersPresent", "c)", "w(", "REPEAT", "R=2", "r(", "IFELSE", "c(", "not", "c(", "leftIsClear", "c)", "c)", "i(", "putMarker", "i)", "ELSE", "e(", "putMarker", "e)", "r)", "turnLeft", "w)", "i)", "turnLeft", "r)", "move", "turnLeft", "turnLeft", "m)"]
{"examples": 
	[
		{
			"actions": ["move", "putMarker"], 
			"example_index": 0, 
			"inpgrid_json": 
			{
				"blocked": "", 
				"cols": 3, 
				"crashed": false, 
				"hero": "11:0:east", "markers": "2:0:1 0:1:8", "rows": 14
			}, 
			"outgrid_json": 
			{
				"blocked": "", 
				"cols": 3, "crashed": false, "hero": "11:1:east", "markers": "11:1:1 2:0:1 0:1:8", "rows": 14
			}, 
		},
	] 
*)
