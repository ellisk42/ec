(* I hate Core *)

open Unix

exception Timeout;;
let sigalrm_handler = Sys.Signal_handle (fun _ -> raise Timeout);;
let run_for_interval (time : float) (c : unit -> 'a) : 'a option =
  (* Install a new alarm handler *)
  let old_behavior = Sys.signal Sys.sigalrm sigalrm_handler in
   let reset_sigalrm () = Sys.set_signal Sys.sigalrm old_behavior 
  in
  try
    ignore (Unix.setitimer ITIMER_REAL {it_interval = 0.0; it_value = time}) ;
    let res = c () in 
    ignore (Unix.setitimer ITIMER_REAL {it_interval = 0.0; it_value = 0.0}) ;
    reset_sigalrm () ;
    Some(res)
  with | Timeout -> begin
      ignore (Unix.setitimer ITIMER_REAL {it_interval = 0.0; it_value = 0.0}) ; 
      reset_sigalrm ();
      None
    end
       | e -> begin
           ignore (Unix.setitimer ITIMER_REAL {it_interval = 0.0; it_value = 0.0}) ;
           reset_sigalrm ();
           raise e
         end


let run_with_timeout ~timeout:timeout f =
  let open Sys in
  let old_handler = Sys.signal Sys.sigalrm
    (Sys.Signal_handle (fun _ -> raise Timeout)) in
  let finish () =
    ignore (Unix.alarm 0);
    ignore (Sys.signal Sys.sigalrm old_handler) in
  try
    ignore (Unix.alarm timeout);
    ignore (f ());
    finish ()
  with Timeout -> finish ()
   | exn -> finish (); raise exn

(* let _ = *)
(*   run_with_timeout 2 (fun () -> *)
(*       let r = ref [] in *)
(*       while true do *)
(*         r := 1 :: !r *)
(*       done); *)

  
(*   print_string "TIMEOUT";; *)

