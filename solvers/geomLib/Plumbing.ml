open Interpreter
open Renderer

let bigarrayWith0s size =
  let data = Bigarray.(Array1.create int8_unsigned c_layout size) in
  Bigarray.Array1.fill data 0 ;
  data

let canvas_to_tlist size canvas =
  try begin
    canvas_to_1Darray canvas size
  end
  with Invalid_argument _ ->
    bigarrayWith0s size

let relist data = (* SLOWWWWW *)
  let l = ref [] in
  for i = (Bigarray.Array1.dim data) - 1 downto 0 do
    l := data.{i}::(!l)
  done ;
  !l

let nop                   = Nop
let concat p1 p2          = Concat(p1,p2)
let embed p               = Embed(p)
let turn x                = Turn(x)
let define x              = Define("MyOnlyVar",x)
let repeat v p            = Repeat(v,p)
let run ?size:(size=28) p = canvas_to_tlist size (interpret p)

let integrate v1 v2 v3 v4 = Integrate(v1,Some(v2),(None,v3,v4,None))
let basic_line            = Integrate(None,Some(true),(None,None,None,None))

let var_unit       = Unit
let var_two        = Next(Unit)
let var_three      = Next(Next(Unit))
let var_half     v = Half(v)
let var_double   v = Double(v)
let var_next     v = Next(v)
let var_prev     v = Prev(v)
let var_divide v v'= Divide(v,v')
let var_opposite v = Opposite(v)
let var_name       = Name("MyOnlyVar")
