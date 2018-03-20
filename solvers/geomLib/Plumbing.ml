open Interpreter
open Renderer

let size = 16

let bigarrayWith0s () =
  let data = Bigarray.(Array1.create int8_unsigned c_layout size) in
  Bigarray.Array1.fill data 0 ;
  data

let threshold data =
  for i = 0 to (Bigarray.Array1.dim data) - 1 do
    data.{i} <- (if data.{i} = 0 then 0 else 1)
  done ;
  data

let canvas_to_tlist canvas =
  try begin
    canvas_to_1Darray canvas size |>
    threshold
  end
  with Invalid_argument _ ->
    bigarrayWith0s ()

let relist data = (* SLOWWWWW *)
  let l = ref [] in
  for i = (Bigarray.Array1.dim data) - 1 downto 0 do
    l := data.{i}::(!l)
  done ;
  !l


let nop            = Nop
let concat p1 p2   = Concat(p1,p2)
let embed p        = Embed(p)
let turn x         = Turn(x)
let repeat v p     = Repeat(v,p)
let run p          = canvas_to_tlist (interpret p)

let integrate v1 v2 v3 v4 v5 v6 = Integrate(v1,v2,(v3,v4,v5,v6))

let var_unit       = Unit
let var_half     v = Half(v)
let var_double   v = Double(v)
let var_next     v = Next(v)
let var_prev     v = Prev(v)
let var_opposite v = Opposite(v)
