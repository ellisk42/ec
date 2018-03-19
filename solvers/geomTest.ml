open Plotter
open Renderer
open Interpreter
open Printf
open Images

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

let npp s l =
  let rec aux h s l = match l with
   | []     -> ()
   | x :: r ->
       if (x = 0) then print_char '0' else print_char '1' ;
       aux (h+1) s r
  in aux 1 s l
let pp s l =
  let rec aux h s l = match l with
   | []     -> ()
   | x :: r ->
       if (x = 0) then print_char '.' else print_char '#' ;
       if (h mod s) = 0 then begin
         print_newline () ;
         aux 1 s r
       end else aux (h+1) s r
  in aux 1 s l

let () =
  let choice = circle in
  let (path,box) = interpret choice in
  let l = Plumbing.run choice in
  pp 16 l ;
  npp 16 l ;
  print_newline () ;
  output_canvas_png (path,box) 16 "toto.png"

