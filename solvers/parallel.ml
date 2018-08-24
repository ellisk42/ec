open Core

open Client
open Utils

let parallel_do nc actions =
  let finished_actions = ref 0 in
  let number_of_actions = List.length actions in
  
  let children = ref [] in
  let actions = ref actions in

  while !finished_actions < number_of_actions do
    (* Printf.printf "Finished %d/%d\n"
     *   !finished_actions number_of_actions;
     * flush_everything(); *)
    (* spawn processes *)
    while List.length !children < nc && List.length !actions > 0 do
      (* Printf.printf "SPAWN\n"; *)
      flush_everything ();
      let next_action = List.hd_exn !actions in
      actions := List.tl_exn !actions;
      match Unix.fork() with
      | `In_the_child -> (next_action(); exit 0)
      | `In_the_parent(c) -> children := c :: !children
    done;

    (* wait for something to die *)
    let (p,_) = Unix.wait `My_group in
    children := List.filter !children ~f:(fun p' -> not (p = p'));
    (* Printf.printf "DEATH\n";
     *   flush_everything(); *)
    incr finished_actions
  done
;;



(* paralleled map *)
let pmap ?processes:(processes=4) ?bsize:(bsize=0) f input output =
  if processes = 0 then begin
        Printf.printf "WARNING: processes = 0\n"; Out_channel.flush stdout
  end ;
  let bsize = match bsize with
    | 0 -> Array.length output / processes
    | x -> x
  in
  (* Given the starting index of a block, computes ending index *)
  let end_idx start_idx = min ((Array.length output) - 1) (start_idx+bsize-1) in
  let next_idx, total_computed = ref 0, ref 0
  and in_streams = ref []
  in
  Gc.compact();
  while !total_computed < Array.length output do
    (* Spawn processes *)
    while !next_idx < Array.length output && List.length !in_streams < processes do
      let rd, wt = Unix.pipe () in
      match Unix.fork () with
      | `In_the_child -> begin
          (* Child *)
          Unix.close rd;
          let start_idx = !next_idx in
          let answer    = Array.init (end_idx start_idx - start_idx + 1)
              (fun i -> f (input (i+start_idx))) in
          let chan = Unix.out_channel_of_descr wt in
          Marshal.to_channel chan (start_idx, answer) [Marshal.Closures];
          Out_channel.close chan;
          Out_channel.flush stdout;
          exit 0
        end
      | `In_the_parent(pid) -> begin
          (* Parent *)
          Unix.close wt;
          in_streams := (rd,pid)::!in_streams;
          next_idx   := !next_idx + bsize;
        end
    done;
    (* Receive input from processes *)
    let recvs = Unix.select ~read:(List.map !in_streams ~f:fst)
        ~write:[] ~except:[] ~timeout:`Never () in
    List.iter ~f:(fun descr ->
        let chan = Unix.in_channel_of_descr descr in
        let pid = List.Assoc.find_exn ~equal:(=) !in_streams descr
        and start_idx, answer = Marshal.from_channel chan in
        ignore (Unix.waitpid pid);
        In_channel.close chan;
        Array.blit answer 0 output start_idx (Array.length answer);
        total_computed := Array.length answer + !total_computed)
      recvs.read;
    in_streams := List.filter ~f:(fun (stream,_) -> not (List.mem ~equal:(=) recvs.read stream)) !in_streams;
  done;
  output

let parallel_map ~nc l ~f =
  let input_array = Array.of_list l in
  let output_array = Array.create (Array.length input_array) None in
  let output_array = 
    pmap ~processes:(min (Array.length input_array) nc)
      ~bsize:1
      (fun x -> Some(f x)) (Array.get input_array) output_array
  in 
  Out_channel.flush stdout;
  Array.to_list output_array |> List.map ~f:(safe_get_some "parallel_map")

let parallel_work ~nc ?chunk:(chunk=0) ~final actions =
  if nc = 1 then begin
    actions |> List.iter ~f:(fun a -> a());
    [final()]
  end else 
  let chunk = match chunk with
    | 0 -> List.length actions / nc
    | x -> x
  in
  let number_of_actions = List.length actions in
  let finished_actions = ref 0 in
  let remaining_actions = ref actions in
  let in_streams = ref [] in
  let outputs = ref [] in
  let worker_id = ref 0 in

  Gc.compact();

  while !finished_actions < number_of_actions do
    (* Spawn processes *)
    while List.length !remaining_actions > 0 && List.length !in_streams < nc do
      let rd, wt = Unix.pipe () in
      match Unix.fork () with
      | `In_the_child -> begin
          refresh_socket_connections();
          (* Child *)
          Unix.close rd;
          let my_work = List.take !remaining_actions chunk in
          (* let start_time = Unix.time() in *)
          my_work |> List.iter ~f:(fun a -> a());
          let answer = final() in
          (* Printf.printf "Worker %d executed in time %f\n"
           *   (!worker_id) (Unix.time()-.start_time); *)
          close_socket_connections();
          let chan = Unix.out_channel_of_descr wt in
          Marshal.to_channel chan (List.length my_work, answer) [Marshal.Closures];
          Out_channel.close chan;
          Out_channel.flush stdout;
          exit 0
        end
      | `In_the_parent(pid) -> begin
          (* Parent *)
          Unix.close wt;
          in_streams := (rd,pid)::!in_streams;
          remaining_actions := List.drop !remaining_actions chunk;
          worker_id := !worker_id + 1
        end
    done;
    (* Receive input from processes *)
    let recvs = Unix.select ~read:(List.map !in_streams ~f:fst)
        ~write:[] ~except:[] ~timeout:`Never () in
    List.iter ~f:(fun descr ->
        let chan = Unix.in_channel_of_descr descr in
        let pid = List.Assoc.find_exn ~equal:(=) !in_streams descr
        and (newly_completed, answer) = Marshal.from_channel chan in
        ignore (Unix.waitpid pid);
        In_channel.close chan;
        finished_actions := !finished_actions + newly_completed;
        outputs := answer :: !outputs)
      recvs.read;
    in_streams := List.filter ~f:(fun (stream,_) -> not (List.mem ~equal:(=) recvs.read stream)) !in_streams;
  done;
  !outputs
