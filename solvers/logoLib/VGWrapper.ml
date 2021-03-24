open Vg
open Gg
open Colors

type kind = MOVE | LINE | CIRCLE

let d_from_origin = 4.5
let d2 = 2. *. d_from_origin

type canvas = (kind*float*float) list

let new_canvas : unit -> canvas = fun () -> []

let moveto : canvas -> float -> float -> canvas = fun l x y -> (MOVE,x,y)::l
let lineto : canvas -> float -> float -> canvas = fun l x y -> (LINE,x,y)::l
let circle : canvas -> float -> float -> canvas = fun l x y -> (CIRCLE,x,y)::l

let moveto_np c x y = P.sub  (P2.v x y) c
let lineto_np c x y = P.line (P2.v x y) c
let circle_np c x y = (Vg.P.circle (Gg.P2.v x y) 0.1) c

let moveto_p = fun (c,(ox,oy)) x y ->
  (((P.empty >> (P.sub  (P2.v x y))),0.)::c),(x,y)
let lineto_p = fun (c,(ox,oy)) x y ->
  let l = sqrt ((ox-.x)*.(ox-.x) +. (oy-.y)*.(oy-.y)) in
  ((P.empty >> (P.sub  (P2.v ox oy)) >> (P.line (P2.v x y))),l)::c,(x,y)
let circle_p = fun (c,(ox,oy)) x y ->
  ((P.empty >> (P.sub  (P2.v ox oy)) >> (P.circle (Gg.P2.v x y) 0.1)),0.)::c,(x,y)

let rec convert_canvas c = match c with
  | [] -> (P.sub (Gg.P2.v 0. 0.) P.empty)
  | (t,x,y)::r ->
    let cc = convert_canvas r in
      (match t with
      | MOVE -> moveto_np cc x y
      | LINE -> lineto_np cc x y
      | CIRCLE -> circle_np cc x y)

let rec convert_canvas_pretty c = match c with
  | [] -> ([],(d_from_origin,d_from_origin))
  | (t,x,y)::r ->
      let cc = convert_canvas_pretty r in
      (match t with
      | MOVE -> moveto_p cc x y
      | LINE -> lineto_p cc x y
      | CIRCLE -> circle_p cc x y)

let size = Size2.v d2 d2
let view = Box2.v P2.o (Size2.v d2 d2)
let area = `O { P.o with P.width = 0.225 ; P.join = `Round }
let areaPretty = `O { P.o with P.width = 0.225 ; P.join = `Round ; P.cap = `Round}
let black = I.const (Color.v_srgb 0. 0. 0.)
let red = I.const (Color.v_srgb 1. 0. 0.)
let green = I.const (Color.v_srgb 0. 1. 0.)
let blue = I.const (Color.v_srgb 0. 0. 1.)
let colors = [|red ; green ; blue|]

let list_to_image pretty l =
  let rec build_c c aux = match c with
    | [] -> [],aux
    | (x,l)::r ->
      let l = if l = l then l else 0. in
      let (r',a2) = (build_c r (aux +. l)) in
      ((x,aux,l)::r',a2)
  in
  if pretty then begin
    let (c,(_,_)) = convert_canvas_pretty l in
    let (c_with_index,maxl) = build_c c 0. in
    let maxl = maxl +. 1. in
    List.fold_right
      (fun (path,ol,l) img ->
        let i = (ol/.maxl) +. ((ol /. maxl) *. (l /. maxl)) in
        let (r,g,b) = interpolate_color (1., 0.2, 0.8) (0.2, 0.2, 1.) i in
        I.blend img (I.cut ~area:(areaPretty) path (I.const (Color.v_srgb r g b))))
      c_with_index
      (I.const (Color.v_srgb ?a:(Some(0.)) 0.0 0.0 0.0))
  end else begin
    let p = convert_canvas l in
    I.cut ~area p black
  end

let output_canvas_png ?pretty:(pretty=false) c desired fname =
  let image = list_to_image pretty c in
  let res = 1000. *. (float_of_int desired) /. (Gg.Size2.h size) in
  let fmt = `Png (Size2.v res res) in
  let warn w = Vgr.pp_warning Format.err_formatter w in
  let oc = open_out fname in
  let r = Vgr.create ~warn (Vgr_cairo.stored_target fmt) (`Channel oc) in
  ignore (Vgr.render r (`Image (size, view, image))) ;
  ignore (Vgr.render r `End) ;
  close_out oc

let canvas_to_1Darray c desired =
  let image = list_to_image false c in
  let res = (float_of_int desired) /. (Gg.Size2.h size) in
  let w,h = desired,desired in
  let stride = Cairo.Image.(stride_for_width A8 w) in
  let data = Bigarray.(Array1.create int8_unsigned c_layout (stride * h)) in
  let surface = Cairo.Image.(create_for_data8 data A8 ~stride ~w:w ~h:h) in
  let ctx = Cairo.create surface in
  Cairo.scale ctx res res;
  let target = Vgr_cairo.target ctx in
  let warn w = Vgr.pp_warning Format.err_formatter w in
  let r = Vgr.create ~warn target `Other in
  ignore (Vgr.render r (`Image (size, view, image))) ;
  ignore (Vgr.render r `End) ;
  Cairo.Surface.flush surface ;
  Cairo.Surface.finish surface ;
  data

let display ba =
  let n = int_of_float (sqrt (float_of_int (Bigarray.Array1.dim ba))) in
  for i = 0 to (n-1) do
    for j = 0 to (n-1) do
      prerr_string (if ba.{(i*n) + j} = 0 then "░░" else "██")
    done ;
    prerr_newline ()
  done ;
  prerr_newline ()


