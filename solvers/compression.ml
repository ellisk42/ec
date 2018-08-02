open Core

open Utils
open Type
open Program
open Enumeration
open Task
open Grammar

open Eg
open Versions

let build_graph_from_versions ?steps:(steps=3) expressions =
  let v = new_version_table() in
  let version_indices = expressions |> List.map ~f:(incorporate v) in
  let version_children = version_indices |> List.map ~f:(child_spaces v) |> List.concat |>
                         List.dedup_and_sort ~compare:(-) in
  let expanded_children = ref (version_children |> List.map ~f:(fun z -> [z])) in

  List.range 1 (steps + 1) |> List.iter ~f:(fun i ->
      time_it (Printf.sprintf "Expanded version spaces to contain expressions %d beta steps outward" i)
        (fun () -> 
           expanded_children := !expanded_children |> List.map ~f:(fun ss ->
               ss @ [List.last_exn ss |> recursive_inversion v])));

  let g = new_class_graph() in
  let version_classes = Array.create ~len:(version_table_size v) None in
  let rec extract_classes j =
    match version_classes.(j) with
    | Some(ks) ->
      let ks = ks |> List.map ~f:(chase g) |> List.dedup ~compare:compare_class in
      version_classes.(j) <- Some(ks);
      ks
    | None ->
      let ks = match index_table v j with
        | Union(u) -> u |> List.map ~f:extract_classes |> List.concat
        | AbstractSpace(b) -> extract_classes b |> List.map ~f:(abstract_class g)
        | ApplySpace(f,x) ->
          let f = extract_classes f in
          let x = extract_classes x in
          f |> List.map ~f:(fun f' -> x |> List.map ~f:(fun x' -> apply_class g f' x')) |> List.concat 
        | IndexSpace(n) -> [leaf_class g (Index(n))]
        | TerminalSpace(p) -> [leaf_class g p]
        | Void -> []
        | Universe -> assert false
      in
      let ks = ks |> List.map ~f:(chase g) |> List.dedup ~compare:compare_class  in
      version_classes.(j) <- Some(ks);
      ks
  in

  List.range 1 (steps + 1) |> List.iter ~f:(fun i ->
      time_it (Printf.sprintf "Loaded rewrites %d steps outward into the graph" i)
        (fun _ -> 
           !expanded_children |> List.sort ~compare:(fun xs ys ->
               Float.to_int (log_version_size v (List.nth_exn xs i) -.
                             log_version_size v (List.nth_exn ys i))
             ) |> List.iter ~f:(fun ss ->
          match ss with
          | [] -> assert false
          | s :: ss -> 
          match extract_classes s with
          | [leader] ->
            let followers = List.nth_exn ss (i - 1) in
            Printf.printf "%f\t%d\n" (log_version_size v followers)
              (extract_classes followers |> List.length);
            Printf.printf "# eq = %d\n"
              (g.members_of_class |> Hashtbl.length);
              
            flush_everything();
            extract_classes followers |> List.iter ~f:(fun f ->
                
                ignore(make_equivalent g leader f))
          | _ -> assert false)))
  
      
      
  

  
  
  

let _ =
  let p = parse_program "(lambda (fold $0 empty (lambda (lambda (cons (+ (+ 5 5) (+ $1 $1)) $0)))))" |> get_some in
  build_graph_from_versions ~steps:3 [p]
  (* let p' = parse_program "(+ 9 9)" |> get_some in *)
  (* let j = time_it "calculated versions base" (fun () -> p |> incorporate t |> recursive_inversion t |> recursive_inversion t  |> recursive_inversion t) in *)
  (* extract t j |> List.map ~f:(fun r -> *)
  (*     (\* Printf.printf "%s\n\t%s\n" (string_of_program r) *\) *)
  (*     (\*   (beta_normal_form r |> string_of_program); *\) *)
  (*     (\* flush_everything(); *\) *)
  (*     assert ((string_of_program p) = (beta_normal_form r |> string_of_program))); *)
  (* Printf.printf "Enumerated %d version spaces.\n" *)
  (*   (t.i2s.ra_occupancy) *)

