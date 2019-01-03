open Core

open Utils
open Differentiation
open Program
open Type
open Task
    
let tvector = make_ground "vector"
let tobject = make_ground "object"
let tfield = make_ground "field"

type physics_object = {position : variable list; velocity : variable list;
                       mass : variable}

type physics_field =
  | PositionField
  | VelocityField
;;

primitive "get-field" (tobject @> tfield @> tvector)
  (fun o f -> match f with
     | PositionField -> o.position
     | VelocityField -> o.velocity);;
primitive "position" tfield PositionField;;
primitive "velocity" tfield VelocityField;;
primitive "get-position" (tobject @> tvector)
  (fun o -> o.position);;
primitive "get-velocity" (tobject @> tvector)
  (fun o -> o.velocity);;
primitive "mass" (tobject @> treal)
  (fun o -> o.mass);;
primitive "*v" (treal @> tvector @> tvector)
  (fun r v -> v |> List.map ~f:(fun v' -> v'*&r));;
primitive "/v" (tvector @> treal @> tvector)
  (fun v r -> v |> List.map ~f:(fun v' -> v'/&r));;
primitive "+v" (tvector @> tvector @> tvector)
  (fun a b ->
     List.map2_exn a b ~f:(+&));;
primitive "-v" (tvector @> tvector @> tvector)
  (fun a b ->
     List.map2_exn a b ~f:(-&));;
primitive "yhat" tvector [~$0.;~$1.];;
primitive "normalize" (tvector @> tvector)
  (fun v ->
     let l = v |> List.map ~f:square |> List.reduce_exn ~f:(+&) |> square_root in
     v |> List.map ~f:(fun v' -> v'/&l));;
primitive "sq" (treal @> treal) square;;
primitive "dp" (tvector @> tvector @> treal)
  (fun a b -> List.map2_exn a b ~f:( *&) |> List.reduce_exn ~f:(+&));;
primitive "vector-length" (tvector @> treal)
  (fun v -> v |> List.map ~f:(fun x -> x *& x) |> List.reduce_exn ~f:(+&) |> square_root);;

  
      
                       
register_special_task "physics"
  (fun extra ?timeout:(timeout=0.01) name request _ ->

     let open Yojson.Basic.Util in
     let open Timeout in

     let maybe_float name default =
       try
         extra |> member name |> to_float
       with _ -> default
     in
     let maybe_int name default =
       try
         extra |> member name |> to_int
       with _ -> default
     in
     let temperature = maybe_float "temperature" 1. in
     let parameterPenalty = maybe_float "parameterPenalty" 0. in
     let restarts = maybe_int "restarts" 300 in
     let steps = maybe_int "steps" 50 in
     let lr = maybe_float "lr" 0.5 in
     let decay = maybe_float "decay" 0.5 in
     let grow = maybe_float "grow" 1.2 in
     let lossThreshold = try Some(extra |> member "lossThreshold" |> to_float) with _ -> None in
     let clipOutput = try Some(extra |> member "clipOutput" |> to_float) with _ -> None in
     let clipLoss = try Some(extra |> member "clipLoss" |> to_float) with _ -> None in

     let unpack_real j = ~$(j |> to_float) |> magical in
     let unpack_vector j = j |> to_list |> List.map ~f:unpack_real |> magical in
     let unpack_object j =
       magical {position = j |> member "position" |> unpack_vector;
                velocity = j |> member "velocity" |> unpack_vector;
                mass = j |> member "mass" |> unpack_real;}
     in

     let unpack t =
       if t = tvector then unpack_vector else
       if t = tobject then unpack_object else
       if t = treal then unpack_real else assert (false)
     in

     let arguments, return = arguments_and_return_of_type request in
     let examples =
       extra |> member "examples" |> to_list |> List.map ~f:(fun example ->
           match example |> to_list with
           | [xs;y] ->
             (List.map2_exn arguments (xs |> to_list) ~f:unpack,
              unpack return y)
           | _ -> assert (false))
     in

     let loss = polymorphic_sse ~clipOutput ~clipLoss return in
     {name = name; task_type = request;
      log_likelihood = (fun expression ->
          let (p,parameters) = replace_placeholders expression in
          let p = analyze_lazy_evaluation p in
        (* Returns loss list *)
        let rec loop : 'a list -> Differentiation.variable list option = function
          | [] -> Some([])
          | (xs,y) :: e ->
            try
              match run_for_interval
                      timeout
                      (fun () -> run_lazy_analyzed_with_arguments p xs) with
              | None -> None
              | Some (prediction) ->
                match loop e with
                | None -> None
                | Some(later_loss) ->
                  try Some(loss prediction y :: later_loss)
                  with DifferentiableBadShape -> None
            with | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
                 | _                   -> None
        in
        match loop examples with
        | None -> log 0.0
        | Some(l) ->
          let n = List.length examples |> Int.to_float in
          let d = List.length parameters |> Int.to_float in
          let average_loss = (List.reduce_exn l ~f:(+&)) *& (~$ (1. /. n)) in
          let average_loss = restarting_optimize (rprop ~lr ~decay ~grow)
              ~attempts:restarts
              ~update:0
              ~iterations:(if List.length parameters = 0 then 0 else steps)
              parameters average_loss
          in
          match lossThreshold with
          | None -> 0. -. d*.parameterPenalty -. n *. average_loss /. temperature
          | Some(t) ->
            if List.for_all l ~f:(fun {data=Some(this_loss)} -> this_loss < t)
            then 0. -. d*.parameterPenalty
            else log 0.)})
          

     
     
