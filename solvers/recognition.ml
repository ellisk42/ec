open Core.Std

open Task
    
let export_task_features (tasks_and_targets : ((task*(float list)) list)) (testing : task list) (f : string) : unit = 
  let open Yojson.Basic.Util in
  let serialize_vector v =
    `List(v |> List.map ~f:(fun f -> `Float(f)))
  in
  let features_of_task task =
    task.task_features |> serialize_vector
  in
  `Assoc(
    [("test", `List(testing |> List.map ~f:features_of_task));
     ("train", `List(tasks_and_targets |> List.map ~f:(fun (task,output) ->
          `Assoc([("features", features_of_task task);
                  ("target", serialize_vector output)]))))])
  |> pretty_to_channel stdout
