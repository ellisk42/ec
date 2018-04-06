(* I hate Core *)

open Unix

exception Timeout

let sigalrm_handler = Sys.Signal_handle (fun _ -> raise Timeout)

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
  with
    | Timeout ->
        begin
          ignore (Unix.setitimer ITIMER_REAL {it_interval = 0.0; it_value = 0.0}) ; 
          reset_sigalrm () ;
          None
        end
    | e ->
        begin
          ignore (Unix.setitimer ITIMER_REAL {it_interval = 0.0; it_value = 0.0}) ;
          reset_sigalrm () ;
          raise e
        end
