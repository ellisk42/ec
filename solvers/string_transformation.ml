open Core.Std

open Re2
    
open Utils
open Type
open Program
open Enumeration
open Task
open Grammar
open EC
    
let load_tasks f =
  let open Yojson.Basic.Util in
  let j = Yojson.Basic.from_file f in
  j |> member "tasks" |> to_list |> List.map ~f:(fun t ->
      (*       Printf.printf "%s\n" (Yojson.Basic.pretty_to_string t); *)
      let name = t |> member "name" |> to_string in
      let ex =
        (t |> member "train" |> to_list)@(t |> member "test" |> to_list) |> 
        List.map ~f:(fun example ->
            let x = example |> member "i" |> to_string in
            let y = example |> member "o" |> to_string in
            (* Printf.printf "%s: %s -> %s\n" name x y; *)
            (x,y))
      in
      supervised_task name (tstring @> tstring) ex)
;;

let primitive_emptyString = primitive "emptyString" tstring "";;
let primitive_uppercase = primitive "uppercase" (tstring @> tstring) String.uppercase;;
let primitive_lowercase = primitive "lowercase" (tstring @> tstring) String.lowercase;;
let primitive_capitalize = primitive "capitalize" (tstring @> tstring) String.capitalize;;
let primitive_concatenate = primitive "concatenate" (tstring @> tstring @> tstring) ( ^ );;
let primitive_constant_strings = [primitive "','" tstring ",";
                                  primitive "'.'" tstring ".";
                                  primitive "'@'" tstring "@";
                                  primitive "' '" tstring " ";
                                  primitive "'<'" tstring "<";
                                  primitive "'>'" tstring ">";
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
let primitive_find_string = primitive "find-string" (tstring @> tstring @> tint)
    (fun pattern target ->
       String.index target (pattern.[0]) |> get_some);;
let primitive_replace = primitive "string-replace" (tstring @> tstring @> tstring @> tstring)
    (fun x y s ->
       if String.length x = 0 then raise (Failure "Replacing empty string") else 
         let rec loop s =
           if String.length s = 0 then s
           else if String.is_prefix s ~prefix:x then
             y ^ (String.drop_prefix s (String.length x) |> loop)
           else
             (String.prefix s 1) ^ (String.drop_prefix s 1 |> loop)
         in loop s);;
let primitive_string_split = primitive "split" (tstring @> tstring @> tlist tstring)
    (fun d x -> String.split ~on:d.[0] x);;
let primitive_string_join = primitive "join" (tstring @> tlist tstring @> tstring)
    (fun d xs -> join ~separator:d xs);;



let _ =
  let n = "stringTransformation.json" in
  let n = "syntheticString.json" in
  let tasks = load_tasks n in
  
  let g = primitive_grammar ([ primitive0; (* primitive1; primitive_n1; *)
                               primitive_increment; primitive_decrement;
                               primitive_emptyString;
                               primitive_uppercase;
                               primitive_lowercase;
                               (* primitive_capitalize; *)
                               primitive_concatenate;
                               primitive_slice_string;
                               primitive_nth_string;
                               primitive_replace;
                               primitive_find_string;
                               primitive_map_string;
                               primitive_string_split;
                               primitive_string_join;
                             ]@primitive_constant_strings)
  in
  (* let capitalize = parse_program "(lambda (slice-string k0 (+1 k0) $0))" |> get_some in *)

  (* Printf.printf "%s" (infer_program_type empty_context [] capitalize |> snd |> string_of_type); *)
  (* Printf.printf "%s\n" (string_of_program capitalize); *)
  (* let v : string = evaluate [] capitalize "dAb9fD" in *)
  (* Printf.printf "%s\n" v; *)
  (* Printf.printf "%f\n" (likelihood_under_grammar g (tstring @> tstring) capitalize); *)
  (* assert false; *)
  exploration_compression tasks g
    ~lambda:0.1
    ~alpha:4.
    ~keepTheBest:1 ~arity:2 10000 5
(* (lambda (map-string $0 (lambda (slice-string (-1 k0) k0 $0)) $0)) *)

