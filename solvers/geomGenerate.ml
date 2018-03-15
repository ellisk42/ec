open Plotter
open Renderer
open Interpreter
open Printf
open Images
open Generator

let _ = Random.self_init ()


let gen_name () =
  string_of_int (Random.int 1048576)  (* This is 2 ** 20 *)

let base = "./generated"

let save ?image:(image=false) p c name cost =
  let s = pp_shapeprogram p in
  let oc =
    open_out
    (Printf.sprintf "%s/%s.LoG" base name) in
  let oc_w =
    open_out
    (Printf.sprintf "%s/%s.cost" base name) in
  Printf.fprintf oc "%s" s ;
  Printf.fprintf oc_w "%d\n" cost ;
  close_out oc ;
  close_out oc_w ;
  if image then begin
    let fname = Printf.sprintf "%s/%s.png" base name in
    output_canvas_png c 16 fname ;
    let fname = Printf.sprintf "%s/%s_HIGH.png" base name in
    output_canvas_png c 64 fname
  end

let () =
  let sup = 5000 in
  let generated = Hashtbl.create (sup / 10) in
  let i = ref 0 in
  while !i < sup do
    let p = generate_random () in
    let cost = costProgram p in
    try begin
      let c = interpret p in
      let l = Plumbing.canvas_to_tlist c in
      try begin
        let c',name = Hashtbl.find generated l in
        if c' < cost then ()
        else begin
          Hashtbl.replace generated l (cost,name) ;
          save p c name cost
        end
      end with Not_found ->
      begin
        let name = gen_name () in
        Hashtbl.add generated l (cost,name) ;
        i := !i + 1 ;
        save ~image:(true) p c name cost
      end
    end with MalformedProgram(s) -> ()
  done

