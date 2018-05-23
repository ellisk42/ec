open Core

open Utils
open Type
open Program
open Enumeration
open Task
open Grammar
open Task

type vector = Vector of int*int

let string_of_vector = function | Vector(x,y) -> "(" ^ string_of_int x ^ "," ^ string_of_int y ^ ")"

type command =
  | Circle of vector
  | Rectangle of vector*vector
  | Line of vector*vector

let canonical_command_list : 'a Core.List.t -> 'a Core.List.t =
  fun l ->
    let l2 =
      List.dedup_and_sort
      ~compare:(fun c1 c2 -> if c1 > c2 then 1 else if c1 = c2 then 0 else -1)
      l in
    List.sort
      ~compare:(fun c1 c2 -> if c1 > c2 then 1 else if c1 = c2 then 0 else -1)
      l2

type cid = C|R|L

let string_of_command = function
  | Circle(v) -> "Circle" ^ string_of_vector v
  | Rectangle(a,b) -> "Rectangle(" ^ string_of_vector a ^ ", " ^ string_of_vector b ^ ")"
  | Line(a,b) -> "Line(" ^ string_of_vector a ^ ", " ^ string_of_vector b ^ ")"

let string_of_trace (t : command list) =
  join ~separator:"; " (t |> List.map ~f:string_of_command)

let tmaybe t = kind "maybe" [t];;
let ttrace = make_ground "trace";;
let tvariable = make_ground "loopVariable";;
let tintercept = make_ground "intercept";;
let tcoefficient = make_ground "coefficient";;
let tcoordinate = make_ground "coordinate";;

let primitive_circle = primitive "circle"
    (tcoordinate @> ttrace)
    (fun v -> [Circle(v)])
let primitive_rectangle = primitive "rectangle"
    (tcoordinate @> tcoordinate @> ttrace)
    (fun v1 v2 -> [Rectangle(v1,v2)]);;
let primitive_linear = primitive "linear"
    (tcoefficient @> tintercept @> tvariable @>
     tcoefficient @> tintercept @> tvariable @>
     tcoordinate)
    (fun cx ix vx cy iy vy -> Vector(cx*vx+ix, cy*vy+iy));;
let primitive_loop = primitive "loop"
    (tint @> tmaybe (tvariable @> ttrace) @> (tvariable @> ttrace) @> ttrace)
    (fun n boundary body ->
       let body = (0--(n-1)) |> List.map ~f:body |> List.concat in
       let boundary = match boundary with
         | None -> []
         | Some(b) -> (0--(n-2)) |> List.map ~f:b |> List.concat 
       in
       boundary@body);;
let primitive_union = primitive "trace-union" (ttrace @> ttrace @> ttrace) (@);;


(* placeholders *)
let primitive_coefficient_placeholder = primitive "COEFFICIENT" tcoefficient None;;
let primitive_intercept_placeholder = primitive "INTERCEPT" tintercept None;;
let primitive_coordinate_placeholder = primitive "COORDINATE" tcoordinate None;;

let latex_primitives = [primitive_circle;primitive_rectangle;
                        primitive_linear;
                        primitive_loop;
                        primitive_nothing;
                        primitive_union;
                        primitive_coefficient_placeholder;
                        primitive_intercept_placeholder;
                        primitive_coordinate_placeholder;
                       primitive3;primitive4;]

type guess = {
  mutable positions: vector list;
  mutable x_intercept: int list;
  mutable y_intercept: int list;
  mutable x_slope: int list;
  mutable y_slope: int list;
}

type guesses = {
  rectangles: guess;
  circles: guess;
  lines: guess;
}

let make_empty_guess () =
  let g() = {positions = []; x_intercept = []; y_intercept = []; x_slope = []; y_slope = [];} in
  {rectangles = g();
   circles = g();
   lines = g();}
    

let score_latex output =
  (* Calculate all of the different possible coefficients/intercepts/coordinates/etc. *)
  let g = make_empty_guess() in

  let pair_guesses s1 s2 =
    match (s1,s2) with
    | (Circle(Vector(x1,y1)), Circle(Vector(x2,y2))) -> begin
        g.circles.x_slope <- (x1 - x2) :: g.circles.x_slope;
        g.circles.x_slope <- (x2 - x1) :: g.circles.x_slope;
        g.circles.y_slope <- (y1 - y2) :: g.circles.y_slope;
        g.circles.y_slope <- (y2 - y1) :: g.circles.y_slope;
      end
    | (Rectangle(Vector(x1,y1),Vector(x2,y2)),Rectangle(Vector(a1,b1),Vector(a2,b2))) -> begin 
        g.rectangles.x_slope <- (x1 - a1) :: g.rectangles.x_slope;
        g.rectangles.x_slope <- (a1 - x1) :: g.rectangles.x_slope;
        g.rectangles.x_slope <- (x2 - a2) :: g.rectangles.x_slope;
        g.rectangles.x_slope <- (a2 - x2) :: g.rectangles.x_slope;

        g.rectangles.y_slope <- (y1 - b1) :: g.rectangles.y_slope;
        g.rectangles.y_slope <- (b1 - y1) :: g.rectangles.y_slope;
        g.rectangles.y_slope <- (y2 - b2) :: g.rectangles.y_slope;
        g.rectangles.y_slope <- (b2 - y2) :: g.rectangles.y_slope;        
       end
    | _ -> ()
  and single_guesses = function
    | Circle(Vector(x1,y1)) -> begin
        g.circles.positions <- Vector(x1,y1) :: g.circles.positions;
        g.circles.x_intercept <- x1 :: g.circles.x_intercept;
        g.circles.y_intercept <- y1 :: g.circles.y_intercept;
      end
    | Line(Vector(x1,y1),Vector(x2,y2)) -> begin
        g.lines.positions <- Vector(x1,y1) :: Vector(x2,y2) :: g.lines.positions;
        g.lines.x_intercept <- x1 :: x2 :: g.lines.x_intercept;
        g.lines.y_intercept <- y1 :: y2 :: g.lines.y_intercept;
      end
    | Rectangle(Vector(x1,y1),Vector(x2,y2)) -> begin
        g.rectangles.positions <- Vector(x1,y1) :: Vector(x2,y2) :: g.rectangles.positions;
        g.rectangles.x_intercept <- x1 :: x2 :: g.rectangles.x_intercept;
        g.rectangles.y_intercept <- y1 :: y2 :: g.rectangles.y_intercept;
      end
  in

  output |> List.iter ~f:single_guesses;
  output |> List.iteri ~f:(fun j c ->
      List.drop output (j+1) |> List.iter ~f:(fun c2 -> pair_guesses c c2));

  (* Now we have calculated the guesses. *)
  g.circles.x_intercept |> List.iter ~f:(fun z -> Printf.eprintf "cxi: %d\t" z);
  g.circles.y_intercept |> List.iter ~f:(Printf.eprintf "cyi: %d\t");
  g.circles.x_slope |> List.iter ~f:(Printf.eprintf "cxm: %d\t");
  g.circles.y_slope |> List.iter ~f:(Printf.eprintf "cym: %d\t");
  Printf.eprintf "\n\n";
  
  let rec random_instantiation ~x ~i expression = match expression with
    | Primitive(t,"COORDINATE",_) ->
      Primitive(t,"COORDINATE",magical @@ ref (random_choice @@ match i with
        |C -> g.circles.positions
        |R -> g.rectangles.positions
        |L -> g.lines.positions))
    | Primitive(t,"INTERCEPT",_) ->
      Primitive(t,"INTERCEPT",magical @@ ref (random_choice @@ match (i,x) with
        |(C,true) -> g.circles.x_intercept
        |(C,false) -> g.circles.y_intercept
        |(L,true) -> g.lines.x_intercept
        |(L,false) -> g.lines.y_intercept
        |(R,true) -> g.rectangles.x_intercept
        |(R,false) -> g.rectangles.y_intercept))
    | Primitive(t,"COEFFICIENT",_) ->
      Primitive(t,"COEFFICIENT",magical @@ ref (random_choice @@ match (i,x) with
        |(C,true) -> g.circles.x_slope
        |(C,false) -> g.circles.y_slope
        |(L,true) -> g.lines.x_slope
        |(L,false) -> g.lines.y_slope
        |(R,true) -> g.rectangles.x_slope
        |(R,false) -> g.rectangles.y_slope))
    | Apply(f,x) ->
      Apply(random_instantiation ~x:(random_choice [true;false]) ~i:(random_choice [C;]) f,
            random_instantiation ~x:(random_choice [true;false]) ~i:(random_choice [C;]) x)
    | Invented(_,b) -> random_instantiation ~x:(random_choice [true;false]) ~i:(random_choice [C]) b
    | Abstraction(b) ->
      Abstraction(random_instantiation ~x:(random_choice [true;false]) ~i:(random_choice [C]) b)
    | anything_else -> anything_else
  in

  let rec likelihood_penalty = function
    | Apply(f,x) -> likelihood_penalty f +. likelihood_penalty x
    | Abstraction(b) -> likelihood_penalty b
    | Invented(_,b) -> likelihood_penalty b
    | Primitive(_,"COORDINATE",_) -> -2.
    | Primitive(_,"INTERCEPT",_) -> -1.
    | Primitive(_,"COEFFICIENT",_) -> -1.
    | _ -> 0.
  in

  let output = canonical_command_list output in

  fun program -> begin      
      if List.exists (0--100) ~f:(fun _ ->
          let p = random_instantiation ~x:false ~i:C program in
          let v : command list = evaluate [] p |> magical |> canonical_command_list in
          v = output)
      then begin Printf.eprintf "PROGRAM: %s\n" (string_of_program program);
        10.*. likelihood_penalty program
      end else log 0.
    end        
        
        
  


let latex_task name output =
  {name = name;
   task_type = ttrace;
   log_likelihood = score_latex output}
  
  
(* let () = *)
(*   let p = parse_program "(loop 3 nothing (lambda (circle (linear COEFFICIENT INTERCEPT $0 COEFFICIENT INTERCEPT $0))))" |> get_some in *)
(*   Printf.printf "%s\n" (string_of_program p); *)
(*   let t = infer_program_type empty_context [] p |> snd in *)
(*   let g = primitive_grammar latex_primitives in *)
(*   Printf.printf "%s\n" (string_of_type t); *)
(*   Printf.printf "likelihood %f\n" @@ score_latex [Circle(Vector(1,2));Circle(Vector(1,4));Circle(Vector(1,6))] p; *)
(*   Printf.printf "log prior %f\n" @@ likelihood_under_grammar g t p; *)
(*   () *)
