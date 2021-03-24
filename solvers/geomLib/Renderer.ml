open Vg
open Gg
open Plotter
open Cairo

type canvas = Plotter.canvas

let output_canvas_png : ?smart:bool ->
                        canvas -> int -> string -> unit =
    fun ?smart:(smart=false) (canvas, box) desired fname ->
    let (view,size,image) = Utils2.get_infos smart d_from_origin box canvas in
    let res = (float_of_int desired) /. (Gg.Size2.h size) in
    let rel = 1000. *. res in (* Sorry, for some reason the unit changes here
                                 from mm to m...! *)
    let fmt = `Png (Size2.v rel rel) in
    let warn w = Vgr.pp_warning Format.err_formatter w in
    let oc = open_out fname in
    let r = Vgr.create ~warn (Vgr_cairo.stored_target fmt) (`Channel oc) in
    ignore (Vgr.render r (`Image (size, view, image))) ;
    ignore (Vgr.render r `End) ;
    close_out oc

let canvas_to_1Darray =
    fun ?smart:(smart=false) (canvas, box) desired ->
    let (view,size,image) = Utils2.get_infos smart d_from_origin box canvas in
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
