open Unix

open Program

let _ =
  let idRef = Sys.argv.(1)
  and p    = Sys.argv.(2) in
  let s_in, s_out =
    open_connection
    (ADDR_UNIX("./prototypical-networks/protonet_socket")) in
  let p = match parse_program p with | Some(p) -> p | _ -> failwith "NOP" in
  let p = analyze_lazy_evaluation p in
  let x = run_lazy_analyzed_with_arguments p [] in
  let l = LogoLib.LogoInterpreter.turtle_to_array x 28 in
  let bytes_version = Bytes.create (28 * 28) in
  for i = 0 to (28 * 28) - 1 do
    Bytes.set bytes_version i (char_of_int (l.{i}))
  done ;
  let img = Bytes.to_string bytes_version in
  output_binary_int s_out (String.length idRef) ;
  output_string s_out idRef ;
  output_binary_int s_out (String.length img) ;
  output_string s_out img ;
  flush s_out ;
  let l = input_binary_int s_in in
  let log_likelihood = (float_of_string (really_input_string s_in l)) in
  output_binary_int s_out (String.length "DONE") ;
  output_string s_out "DONE" ;
  flush s_out ;
  shutdown_connection s_in ;
  close_in s_in ;
  print_endline (
    Printf.sprintf "Raw dist:\t%f\nProposal:\t%f"
    log_likelihood (-. (100. *. log_likelihood))
  )

