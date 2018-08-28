open Core

open Zmq

type socket_connection = (([`Req] Zmq.Socket.t) ref * int) ;;
let context = ref (Zmq.Context.create());;
let socket_connections : (socket_connection list) ref = ref [];;

let socket_json (socket : socket_connection) message = 
  let open Yojson.Safe in
  let message = to_string message in
  let socket,_ = socket in
  Zmq.Socket.send !socket message;
  let response = Zmq.Socket.recv !socket in
  response |> from_string

let connect_socket' p : [`Req] Zmq.Socket.t = 
  let socket = Zmq.Socket.create !context Zmq.Socket.req in
  Zmq.Socket.connect socket @@ "tcp://localhost:"^(p |> Int.to_string);
  socket

let connect_socket p : socket_connection =
  let s = connect_socket' p in
  let s = (ref s,p) in
  socket_connections := s :: !socket_connections;
  s

let refresh_socket_connections() =
  match !socket_connections with
    [] -> ()
  | _ -> begin
      context := Zmq.Context.create();
      socket_connections := !socket_connections |> List.map ~f:(fun (r,p) ->
          r := connect_socket' p;
          (r,p));
    end

let close_socket_connections() =
  !socket_connections |> List.iter ~f:(fun (r,p) ->
      Zmq.Socket.close !r);
  socket_connections := []

(* let rec test socket = *)
(*   Zmq.Socket.send socket message; *)
(*   let response = Zmq.Socket.recv socket in *)
(*   assert (response = message); *)
(*   Printf.printf "Client received %s\n" response; *)
(*   Pervasives.flush stdout; *)
(*   test socket;; *)
(* let () = *)
  (* let socket = Zmq.Socket.create context Zmq.Socket.req in *)
  (* Zmq.Socket.connect socket "tcp://localhost:9119"; *)
(*   test socket  *)
(* ;; *)
