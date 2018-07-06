open Core

open Timeout
open Task
open Utils
open Program
open Type

let ttower = make_ground "tower";;

type tower_result = {stability : float;
                     length : float;
                     area : float;
                     staircase : float;
                     height : float;
                     overpass : float;}

let center_tower p =
  let xs = p |> List.map ~f:(fun (x,_,_) -> x) in
  let x1 = List.fold_left ~init:(List.hd_exn xs) ~f:max xs in
  let x0 = List.fold_left ~init:(List.hd_exn xs) ~f:min xs in
  let c = (x1-.x0)/.2. in
  p |> List.map ~f:(fun (x,w,h) -> (x-.c,w,h))

let block w h =
  let n = Printf.sprintf "%dx%d" w h in
  let xOffset = if w mod 2 = 1 then 0.5 else 0.0 in
  let e = 0.05 in
  let w = Float.of_int w -.e in
  let h = Float.of_int h -.e in
  let v = fun k -> (xOffset,w,h) :: k in
  ignore(primitive n (ttower @> ttower) v)
;;


block 3 1;;
block 1 3;;
block 1 1;;
block 2 1;;
block 1 2;;
block 4 1;;
block 1 4;;

ignore(primitive "left" (ttower @> ttower) (fun t -> t |> List.map ~f:(fun (x,w,h) -> (x-.1.0,w,h))));;
ignore(primitive "right" (ttower @> ttower) (fun t -> t |> List.map ~f:(fun (x,w,h) -> (x+.1.0,w,h))));;


let connection_failures = ref 0;;
let connection_successes = ref 0;;

let send_to_tower_server k =
  let open Yojson.Safe in
  let p = 9494 in
  let h = Unix.Inet_addr.localhost in
  let (ic,oc) =
    let attempts = ref 0 in
    let connection = ref None in
    while
      try
        connection := Some(Unix.open_connection (ADDR_INET(h,p)));
        if !attempts > 0
        then begin 
          connection_failures := !connection_failures + 1;
          Printf.eprintf "Connected to socket after %d attempts (%d/%d attempts had problems)\n" (!attempts) (!connection_failures) (!connection_failures + !connection_successes);
          flush_everything()
        end else connection_successes := !connection_successes + 1
        ;
        false
      with Unix.Unix_error(_,_,_) -> true
    do
      attempts := !attempts + 1;
      Thread.delay 0.05
    done;
    !connection |> get_some        
  in
  (*   let message : string = Yojson.Basic.to_string k in *)
  (* output_string oc message;
   * output_string oc "\n"; *)
  Yojson.Safe.to_channel oc k;
  (* basin *)
  Out_channel.flush oc;
  (* Printf.printf "outputted message %s\n" message; *)
  Unix.shutdown ~mode:SHUTDOWN_SEND (Unix.descr_of_out_channel oc);
  
  let r = Yojson.Safe.from_channel ic in
  (*   Unix.shutdown_connection  *)
  Unix.shutdown ~mode:SHUTDOWN_RECEIVE (Unix.descr_of_in_channel ic);
  In_channel.close ic;
  (*   Printf.printf "gotTheMessage %s\n" (pretty_to_string r); *)
  flush_everything();
  r

let parse_tower_result result =
  let open Yojson.Safe.Util in
  let stability = result |> member "stability" |> to_float in
  let length = result |> member "length" |> to_float in
  let area = result |> member "area" |> to_float in
  let staircase = result |> member "staircase" |> to_float in
  let height = result |> member "height" |> to_float in
  let overpass = result |> member "overpass" |> to_float in
  {stability;length;area;staircase;height;overpass;}  

let tower_cash = Hashtbl.Poly.create();;



let update_tower_cash() =
  let open Yojson.Safe.Util in
  let open Yojson.Safe in
  let m = (`String("sendCash")) in
  let filename =  send_to_tower_server m |> to_string in
  (* For some reason the filename still includes quotes, so we need to remove the first and last characters *)
  let filename = String.sub ~pos:1 ~len:(String.length filename - 2) filename in
  let new_entries = Yojson.Safe.from_file filename in
  
  new_entries |> to_list |> List.iter ~f:(fun e ->
      match e |> to_list with
      | [`List([plan;perturbation;]);result] ->
        let plan = plan |> to_list |> List.map ~f:(fun b -> match b |> to_list with
            | [a;b;c] -> (a |> to_float, b |> to_float, c |> to_float)
            | _ -> raise (Failure ("plan entry\n"^(pretty_to_string plan)))) in
        let perturbation = perturbation |> to_float in
        Hashtbl.Poly.set tower_cash ~key:(plan,perturbation) ~data:(parse_tower_result(result))
      | _ -> raise (Failure "tower cache entry"))
  ;
  Sys.remove filename


let evaluate_tower ?n:(n=15) plan perturbation =
  (* center the tower *)
  let plan = center_tower plan in
  let key = (plan, perturbation) in
  let open Yojson.Safe.Util in
  match Hashtbl.Poly.find tower_cash key with
  | Some(r) -> begin
      (* Printf.eprintf "(ocaml: hit %s)\n" *)
      (*   (plan |> List.map ~f:(fun (a,b,c) -> Printf.sprintf "%f,%f,%f" a b c) |> join ~separator:"; "); *)
      (* flush_everything(); *)
      r
    end
  | None ->
    (* Printf.eprintf "(ocaml: miss %s)\n" *)
    (*   (plan |> List.map ~f:(fun (a,b,c) -> Printf.sprintf "%f,%f,%f" a b c) |> join ~separator:"; "); *)
    (* flush_everything(); *)
      
    let plan = `List(plan |> List.map ~f:(fun (a,b,c) -> `List([`Float(a);
                                                                `Float(b);
                                                                `Float(c);]))) in
    let response = send_to_tower_server (`Assoc([("plan",plan);
                                                 ("n",`Int(n));
                                                 ("perturbation",`Float(perturbation))])) in
    let r = parse_tower_result response in
    Hashtbl.Poly.set tower_cash ~key:key ~data:r;
    r

let tower_task ?timeout:(timeout = 0.001)
    ?stabilityThreshold:(stabilityThreshold=0.5)
    ~perturbation ~maximumStaircase ~maximumMass ~minimumLength ~minimumArea ~minimumHeight ~minimumOverpass
    name task_type examples =
  assert (task_type = ttower @> ttower);
  assert (examples = []);
  { name = name    ;
    task_type = task_type ;
    log_likelihood =
      (fun p ->
         let p = analyze_lazy_evaluation p in
         try
           match run_for_interval
                   timeout
                   (fun () -> run_lazy_analyzed_with_arguments p [[]])
           with
           | Some(p) ->
             let m = p |> List.map ~f:(fun (_,w,h) -> w*.h) |> List.fold_right ~f:(+.) ~init:0. in
             if m > maximumMass then log 0.0 else 
               let result = evaluate_tower p perturbation in
               if result.height < minimumHeight
               || result.staircase > maximumStaircase
               || result.stability < stabilityThreshold
               || result.length < minimumLength
               || result.area < minimumArea
               || result.overpass < minimumOverpass
               then log 0.0
               else 50.0 *. log result.stability
           | _ -> log 0.0
         with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
              (* we have to be a bit careful with exceptions *)
              (* if the synthesized program generated an exception, then we just terminate w/ false *)
              (* but if the enumeration timeout was triggered during program evaluation, we need to pass the exception on *)
              | otherException -> begin
                  if otherException = EnumerationTimeout then raise EnumerationTimeout else log 0.
                end)

  }

(* let test_tower() = *)
(*   update_tower_cash(); *)
(*   let t = tower_task ~perturbation:4. ~maximumStaircase:10. ~maximumMass:200. *)
(*       ~minimumLength:1.0 ~minimumArea:0. ~minimumHeight:2. "test" ttower [] in *)
(*   let p = parse_program "(do 3x1 (do (left 1x3) (do (right 1x3) 3x1)))" |> get_some in *)
(*   Printf.printf "%f\n" (t.log_likelihood p) *)
(*   ; *)
(*   flush_everything() *)
