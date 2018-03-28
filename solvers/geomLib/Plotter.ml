open Vg
open Gg

type canvas = P.t * Gg.Box2.t

let d_from_origin = 1.

let new_canvas () = (P.sub (Gg.P2.v d_from_origin d_from_origin) P.empty,
                     Gg.Box2.empty)

let middle_x = fun _ -> d_from_origin
let middle_y = fun _ -> d_from_origin

let moveto : canvas -> float -> float -> canvas =
    fun (path, box) x y ->
    let new_path = P.sub (P2.v x y) path in
    (new_path, box)

let lineto : canvas -> float -> float -> canvas =
    fun (path, box) x y  ->
    let new_path = P.line (P2.v x y) path in
    (new_path, Gg.Box2.add_pt box (Gg.P2.v x y))

let print_canvas canvas =
    let (canvas,_) = canvas in
    print_endline (P.to_string canvas)
