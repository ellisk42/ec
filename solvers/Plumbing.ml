open Interpreter
open Renderer

let size = 16

let canvas_to_tlist canvas =
  let l = try canvas_to_1Darray canvas size
          with Invalid_argument _ -> List.init (size*size) (fun _ -> 1) in
  List.map (fun x -> if (x = 0) then 0 else 1) l

let nop            = Nop
let concat p1 p2   = Concat(p1,p2)
let embed p        = Embed(p)
let turn x         = Turn(x)
let repeat v p       = Repeat(v,p)
let run p          = canvas_to_tlist (interpret p)

let integrate v1 v2 v3 v4 v5 v6 = Integrate(v1,v2,(v3,v4,v5,v6))

let var_unit       = Unit
let var_half     v = Half(v)
let var_double   v = Double(v)
let var_next     v = Next(v)
let var_prev     v = Prev(v)
let var_opposite v = Opposite(v)
(*let var_name s     = Name(s)*)
