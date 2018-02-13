open Core.Std

open Utils
open Type
open Program
open Enumeration
open Task
open Grammar
open Compression
open EC

let maximumCoefficient = 9
  
let polynomial_tasks =
  (0--maximumCoefficient) |> List.map ~f:(fun a ->
      (0--maximumCoefficient) |> List.map ~f:(fun b ->
          (0--maximumCoefficient) |> List.map ~f:(fun c ->
              let examples = List.map (0--5) ~f:(fun x -> (x, a*x*x + b*x + c)) in
              let n = Printf.sprintf "(%i x^2 + %i x + %i)" a b c in
              supervised_task n (tint @> tint) examples)))
  |> List.concat |> List.concat

let polynomial_grammar =
  primitive_grammar [ primitive0;
                      primitive1;
                      (* primitive2; *)
                      (* primitive3; *)
                      (* primitive4; *)
                      (* primitive5; *)
                      (* primitive6; *)
                      (* primitive7; *)
                      (* primitive8; *)
                      (* primitive9; *)
                      primitive_addition;
                      primitive_multiplication;
                      (* primitive_apply; *)
                    ]


                                                             
let _ =
  exploration_compression polynomial_tasks polynomial_grammar ~keepTheBest:3 10000 1 ~alpha:10.
