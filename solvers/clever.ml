open Core.Std

open Utils

type tree =
  | T of string*(tree list)

let rec show = function
  | T(n,[]) -> n
  | T(n,xs) -> "("^n^" "^(join ~separator:" " (xs |> List.map ~f:show))^")"

let productions = [("x",0);
                   ("f",1);
                   ("g",2)]

type entry =
  | Finished of tree
  | NotDone of (unit -> unit)

(* let entry_size = function *)
(*   | Finished(s,_) -> s *)
(*   | NotDone(s,_) -> s *)

let _ =
  let pq : (int*entry) list ref = ref [] in
  let choice parentCost xs =
    pq := !pq @ (xs |> List.map ~f:(fun (s,x) -> (s+parentCost,NotDone(x))))
  in
  let rec g (parentCost : int) (k : int*tree -> unit) : unit =
    choice parentCost (productions |> List.map ~f:(fun (f,a) ->
        (1,
         fun () ->
           ga (parentCost+1) a (fun (newCost, arguments) ->
               (newCost, T(f,arguments)) |> k))))
  and ga (parentCost : int) (n : int) k =
    if n = 0 then k (parentCost,[]) else
      g parentCost (fun (cost1,argument1) ->
          ga cost1 (n-1) (fun (cost2,suffix) ->
              (cost2, argument1::suffix) |> k))
  in
  g 0 (fun (size, tree) -> pq := !pq @ [(size,Finished(tree))]);

  (0--10000) |> List.iter ~f:(fun j ->
      (* Printf.printf "Iteration %d, |pq| = %d\n" j (List.length !pq); *)
      match List.sort ~cmp:(fun (s1,_) (s2,_) -> s1 - s2) !pq with
      | [] -> Printf.printf "Empty priority queue\n"
      | (cost,Finished(t))::suffix -> begin 
          Printf.printf "Produced cost %d tree %s\n" cost (show t);
          pq := suffix
        end
      | (_,NotDone(k))::suffix -> begin 
          (* Printf.printf "Invoking callback...\n"; *)
          pq := suffix;
          k ()
        end
    )

      
                             


(* let apply_builder (builder : tree -> entry) (builder_size : int) () =  *)

(* let rec expand () : entry list = *)
(*   [Finished(1,Terminal); *)
(*    NotDone(1, expand_branch)] *)
(* and expand_branch () : entry list =  *)
(*   (\* We have size one, and if an extension is requested then we first expand the left child *\) *)
(*   (\* For now assume that the right child is always terminal *\) *)
(*   let ls = expand () in *)
(*   ls |> List.map ~f:(function *)
(*       | NotDone(left_size, left_expander) ->  *)
(*         NotDone(1 + left_size, fun () -> *)
(*             left_expander () |>  *)
          
(*       | Finished(left_size, left_tree) -> *)
(*         [Finished(1 + left_size, Branch(left_tree, Terminal))] *)
(*     ) |> List.concat *)
