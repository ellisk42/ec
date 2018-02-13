open Core.Std

open Utils
open Type
open Program
open Enumeration
open Task
open Grammar
open Task


let load_list_tasks f =
  let open Yojson.Basic.Util in
  let j = Yojson.Basic.from_file f in
  j |> to_list |> List.map ~f:(fun t ->
      (*       Printf.printf "%s\n" (Yojson.Basic.pretty_to_string t); *)
      let name = t |> member "name" |> to_string in
      (* change if needed *)
      let returnType = ref tint in
      let ex =
        t |> member "examples" |> to_list |>
         List.map ~f:(fun example ->
            let [x;y] = to_list example in
            (x |> to_list |> List.map ~f:to_int,
             try
               y |> to_int |> magical
             with _ ->
               begin
                  returnType := tlist tint;
                  y |> to_list |> List.map ~f:to_int |> magical
                end))
      in
      supervised_task name (tlist tint @> !returnType) ex)


let list_grammar =
  primitive_grammar [ primitive0;
                      primitive1;
                      primitive2;
                      primitive3;
                      primitive4;
                      primitive5;
                      (* primitive6; *)
                      (* primitive7; *)
                      (* primitive8; *)
                      (* primitive9; *)
                      primitive_addition;
                      (* primitive_subtraction; *)
                      primitive_sort;
                      (* primitive_reverse; *)
                      (* primitive_append; *)
                      primitive_empty;
                      (* primitive_singleton; *)
                      (* primitive_slice; *)
                      (* primitive_length; *)
                      primitive_map;
                      primitive_reducei;
                      (* primitive_filter; *)
                      primitive_equal;
                      primitive_not;
                      primitive_if;
                      primitive_cons;
                      primitive_is_square;
                      primitive_greater_than;]

let _ =
  let t = supervised_task "filter-squares" (tlist tint @> tlist tint)
      [([1;2;1;9;4;3;2],[1;1;9;4])] 
  in
  enumerate_for_task ~timeout:30000 list_grammar t


(* let _ = *)
(*   exploration_compression (load_list_tasks "list_tasks.json") *)
(*     list_grammar *)
(*     ~keepTheBest:1 ~arity:1 *)
(*     1000000 (\*frontier size*\) *)
(*     5 (\*iterations*\) *)
