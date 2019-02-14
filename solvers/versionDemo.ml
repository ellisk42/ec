open Core
open Versions
open Program
open Utils

let _ =
  let t = new_version_table() in
  let p = "(#(lambda (lambda (* $2 (+ (lambda $2) $0)))) $0 2)" |> parse_program |> get_some in
  p |> incorporate t |> inline t |> extract t |> List.iter ~f:(fun p' ->
      Printf.printf "%s\n"
        (string_of_program p'));
  assert (false)
;;


let _ =
  List.range 0 6 |> List.iter ~f:(fun sz ->
      let p0 = List.range 0 sz |>
               List.fold_right ~init:"(+ 1 1)" ~f:(fun _ -> Printf.sprintf "(+ 1 %s)") |>
               parse_program |> get_some
      in

      List.range 1 4 |> List.iter ~f:(fun a ->
          let v = new_version_table() in
          let j = incorporate v p0 in
          let r = List.range 0 a |>
                  List.fold_right ~init:[j] ~f:(fun _ (a :: b) -> recursive_inversion v a :: a :: b) |>
                  union v
          in

          let version_size : int = reachable_versions v [r] |> List.length in

          let ht = new_version_table() in
          let distinct_programs = extract v r |> List.map ~f:(incorporate ht) |> List.dedup_and_sort ~compare:(-) |> List.length in
          let program_memory = version_table_size ht in

          Printf.printf "version_size[%d,%d] = %d\n"
            sz a version_size;
          Printf.printf "distinct_programs[%d,%d] = %d\n"
            sz a distinct_programs;
          Printf.printf "program_memory[%d,%d] = %d\n"
            sz a program_memory;
          (* Printf.printf "approximate size = %f\n" *)
          (*   (unique_space v r |> log_version_size v |> exp); *)
          flush_everything()
        ))
          
  
