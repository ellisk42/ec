open Unix

(*
 * TODO : string are not as good as bytes
 *)

let str = "behaviour/404_squareOfCircle"

let gen_test () =
  let data = Bigarray.(Array1.create int8_unsigned c_layout (28 * 28)) in
  Bigarray.Array1.fill data 128 ;
  let bytes_version = Bytes.create (28 * 28) in
  for i = 0 to (28 * 28) - 1 do
    Bytes.set bytes_version i (char_of_int (data.{i}))
  done ;
  Bytes.to_string bytes_version



let _ =
  let s_in, s_out = open_connection (ADDR_UNIX("./protonet_socket")) in
  let str2 = gen_test () in
  output_binary_int s_out (String.length str) ;
  output_string s_out str ;
  output_binary_int s_out (String.length str2) ;
  output_string s_out str2 ;
  flush s_out ;
  let l = input_binary_int s_in in
  let dist = really_input_string s_in l in
  prerr_endline dist ;
  output_binary_int s_out (String.length str) ;
  output_string s_out str ;
  output_binary_int s_out (String.length str2) ;
  output_string s_out str2 ;
  flush s_out ;
  let l = input_binary_int s_in in
  let dist = really_input_string s_in l in
  prerr_endline dist ;
  output_binary_int s_out (String.length "DONE") ;
  output_string s_out "DONE" ;
  flush s_out ;
  shutdown_connection s_in ;
  close_in s_in

