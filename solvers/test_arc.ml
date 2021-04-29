open Core
open Arc

let convert_raw_to_block raw = 
  let open Yojson.Basic.Util in

  let y_length = List.length (raw |> to_list) -1 in
  let x_length = List.length (List.nth_exn (raw |> to_list) 0 |> to_list) - 1 in
  let indices = List.cartesian_product (0 -- y_length) (0 -- x_length) in
  let match_row row x = match List.nth row x with
        | Some c -> c |> to_int
        | None -> (-1) in
  let deduce_val (y,x) = match (List.nth (raw |> to_list) y) with
      | Some row -> match_row (to_list row) x 
      | None -> (-1) in
  let new_points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y,x))) indices in
  {points = new_points; original_grid = new_points} ;;


let test_example assoc_list p = 
  let open Yojson.Basic.Util in
  let raw_input = List.Assoc.find_exn assoc_list "input" ~equal:(=) in
  let raw_expected_output = List.Assoc.find_exn assoc_list "output" ~equal:(=) in
  let input = convert_raw_to_block raw_input in
  let expected_output = convert_raw_to_block raw_expected_output in
  let got_output = p input in
  let matched = got_output === expected_output in
  printf "\n%B\n" matched;
  match matched with 
  | false -> 
    printf "\n Input \n";
    print_block input ;
    printf "\n Resulting Output \n";
    print_block got_output;
    printf "\n Expected Output \n";
    print_block expected_output;
    matched
  | true -> matched;;

let test_task file_name p =
  printf "\n ----------------------------- Task: %s --------------------------- \n" file_name;
  let fullpath = String.concat ["/Users/theo/Development/program_induction/ec/arc-data/data/training/"; file_name; ".json"] in
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let json = from_file fullpath in
  let json = json |> member "train" |> to_list in
  let pair_list = List.map json ~f:(fun pair -> pair |> to_assoc) in 
  try
    let ex_results = List.map pair_list ~f:(fun assoc_list -> test_example assoc_list p) in 
    List.reduce_exn ex_results ~f:(fun prev el -> (prev && el))
  with
    | _ -> printf "\n error in executing program \n"; false in


let p_72ca375d grid = 
  let blocks = find_same_color_blocks grid true false in
  let filtered_blocks = List.filter blocks ~f:(fun block -> is_symmetrical false block) in
  let merged_block = merge_blocks filtered_blocks in
  to_min_grid merged_block false in
(* test_task "72ca375d" p_72ca375d ;; *)


let p_5521c0d9 grid = 
  let blocks = find_same_color_blocks grid true false in
  let get_height block = ((get_max_y block) - (get_min_y block)) in
  let shifted_blocks = map_blocks (fun block -> move block (get_height block) 0) blocks in
  let merged_blocks = merge_blocks shifted_blocks in
  to_original_grid_overlay merged_blocks false false in
test_task "5521c0d9" (fun a -> a);;




















