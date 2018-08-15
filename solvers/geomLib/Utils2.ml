open Vg
open Gg
open Plotter

(* Some utils values and functions *)

let pi = 4. *. atan(1.)
let pi2 = 2. *. pi
let pi4 = 4. *. pi
let pis2 = pi /. 2.
let pis4 = pi /. 4.

let my_print_float f = match f with
 | f when f = pi4  -> "4*π"
 | f when f = pi2  -> "2*π"
 | f when f = pi  -> "π"
 | f when f = pis2 -> "π/2"
 | f when f = pis4 -> "π/4"
 | f when f = (-1.) *. pi4  -> "-4*π"
 | f when f = (-1.) *. pi2  -> "-2*π"
 | f when f = (-1.) *. pi  -> "-π"
 | f when f = (-1.) *. pis2 -> "-π/2"
 | f when f = (-1.) *. pis4 -> "-π/4"
 | _ -> Printf.sprintf "%.4g" f

let my_print_bool b = match b with
 | true -> "on"
 | false -> "off"

let normal_random () =
    let u1 = Random.float 1. and u2 = Random.float 1. in
    (sqrt ((-2.) *. (log u1) )) *. (cos (2. *. pi *. u2))

let pp_view s = Printf.printf "(%f,%f)\n" (Gg.Box2.w s) (Gg.Box2.h s)
let pp_size s = Printf.printf "(%f,%f)\n" (Size2.w s)   (Size2.h s)
let pp_p2 s   = Printf.printf "(%f,%f)\n" (V2.x s)   (V2.y s)

let get_infos smart d_from_origin box canvas =
    let d2 = 2. *. d_from_origin in
    let view_crop =
        try (Gg.Box2.inter
                box
                (Gg.Box2.v (Gg.P2.v 0. 0.) (Gg.Size2.v d2 d2)))
        with Invalid_argument _  -> Gg.Box2.empty
    in
    let s = try Gg.Box2.size view_crop
            with Invalid_argument _ -> Gg.Size2.v 0. 0. in
    let o = try Gg.Box2.o view_crop
            with Invalid_argument _ -> Gg.P2.v d_from_origin d_from_origin in
    let dim = max (Gg.P2.x s) (Gg.P2.y s) in
    let offsetx = (dim -. Gg.P2.x s) /. 2. in
    let offsety = (dim -. Gg.P2.y s) /. 2. in
    let margin = 0.0 in
    let view =
      if smart then
        (Gg.Box2.v
          (Gg.P2.v
            (Gg.P2.x o -. offsetx -. margin)
            (Gg.P2.y o -. offsety -. margin))
        (Gg.P2.v (dim +. (2. *. margin)) (dim +. (2. *. margin))))
      else (Gg.Box2.v (Gg.P2.v 0. 0.) (Gg.Size2.v d2 d2)) in
    let size = Gg.Size2.v d2 d2 in
    let area =
      `O { P.o with
            P.width = if smart then (1. /. dim) else 0.05 ;
            (*P.cap = `Round ; Makes subpath render weird...! *)
            P.join = `Round }
    in
    let black = I.const Color.black in
    let image = I.cut ~area canvas black in
    (view,size,image)
