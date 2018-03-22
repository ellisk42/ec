open GeomLib
open Plotter
open Renderer
open Interpreter
open Printf

(*let prog = (Plumbing.concat Plumbing.integrate*)
              (*(Plumbing.concat (Plumbing.turn None) Plumbing.integrate))*)
(*let prog2 = Plumbing.repeat (Plumbing.repeat prog)*)
let empty  = Turn(None)
let line   = Integrate(None,None,(None,None,None,None))
let angle  = Concat(line,Concat(Turn(None),line))
let square = Repeat(None,Repeat(None,Concat(line,Turn(None))))
let circle = Integrate(None,None,(None,None, Some(Unit),None))
let dashes = Repeat(None,Repeat(None,Concat(line,
Integrate(None,Some(false),(None,None,None,None)))))
let spiral = Integrate(None,None,(None,None,None,Some(Unit)))
(*let spiral = Integrate(None,None,(None,Some(Unit),Some(Unit), None))*)


let pp l data =
  for i = 0 to (Bigarray.Array1.dim data) - 1 do
    if (i mod l) == 0 then print_newline () ;
    print_int (data.{i})
  done

let npp data =
  for i = 0 to (Bigarray.Array1.dim data) - 1 do
    print_int (data.{i})
  done

let () =
  let choice = spiral in
  let (path,box) = interpret choice in
  let l = Plumbing.run choice in
  print_canvas (path,box) ;
  pp 32 l ;
  npp l ;
  print_newline () ;
  output_canvas_png (path,box) 512 "toto.png"

