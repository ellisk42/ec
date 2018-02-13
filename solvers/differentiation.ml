open Core.Std

open Utils
    
type variable = {mutable gradient : float option;
                 mutable data : float option;
                 arguments : variable list;
                 forwardProcedure : float list -> float;
                 backwardProcedure : float list -> float list;
                 (* descendents: every variable that takes this variable as input *)
                 (* the additional float parameter is d Descendent / d This *)
                 mutable descendents : (variable*float) list}

let variable_value v = v.data |> get_some

let make_variable forward backward arguments =
  let values = List.map ~f:(fun v -> v.data |> get_some) arguments in
  let initial_value = forward values in
  let s = {gradient = None;
           arguments = arguments;
           descendents = [];
           forwardProcedure = forward;
           backwardProcedure = backward;
           data = Some(initial_value);}
  in
  let initial_partials = backward values in
  List.iter2_exn initial_partials arguments ~f:(fun p a ->
      a.descendents <- (s,p)::a.descendents);
  s

let make_binary_variable forward backward a b =
  make_variable (fun x -> match x with
      |[a;b] -> forward a b
      |_ -> raise (Failure "Binary variable did not get 2 arguments"))
    (fun x -> match x with
      |[a;b] -> backward a b
      |_ -> raise (Failure "Binary variable did not get 2 backward arguments"))
    [a;b;]

let make_unitary_variable forward backward a =
  make_variable (fun x -> match x with
      |[a;] -> forward a 
      |_ -> raise (Failure "unitary variable did not get 1 arguments"))
    (fun x -> match x with
      |[a;] -> backward a 
      |_ -> raise (Failure "urinary variable did not get 1 backward arguments"))
    [a;]

let (~$) x =
  let rec s = {gradient = None;
               arguments = [];
               descendents = [];
               data = Some(x);
               forwardProcedure = (fun _ -> s.data |> get_some);
               backwardProcedure = (fun _ -> []);
              }
  in s

let placeholder () =
  let rec s = {gradient = None;
               arguments = [];
               descendents = [];
               data = None;
               forwardProcedure = (fun _ -> s.data |> get_some);
               backwardProcedure = (fun _ -> []);
              }
  in s

let random_variable ?mean:(mean = 0.) ?standard_deviation:(standard_deviation = 0.1) () =
  let rec s = {gradient = None;
               arguments = [];
               descendents = [];
               data = Some(normal standard_deviation mean);
               forwardProcedure = (fun _ -> s.data |> get_some);
               backwardProcedure = (fun _ -> []);
              }
  in s

let update_variable v x =
  assert (v.arguments = []);
  v.data <- Some(x)


let (+&) =
  make_binary_variable (+.) (fun _ _ -> [1.;1.]) 

let (-&) =
  make_binary_variable (-.) (fun _ _ -> [1.;-1.]) 

let ( *& ) =
  make_binary_variable ( *. ) (fun a b -> [b;a]) 

let logarithm =
  make_unitary_variable log (fun a -> [1./.a])

let exponential =
  make_unitary_variable exp (fun a -> [exp a])

let log_soft_max xs =
  make_variable (fun vs -> 
      let m : float = List.fold_right vs ~init:Float.neg_infinity  ~f:max in
      let zm = List.fold_right ~init:0. ~f:(fun x a -> exp (x -. m) +. a) vs in
      m+. (log zm))
    (fun vs -> 
      let m : float = List.fold_right vs ~init:Float.neg_infinity  ~f:max in
      let zm = List.fold_right ~init:0. ~f:(fun x a -> exp (x -. m) +. a) vs in
      List.map vs ~f:(fun x -> (exp (x-.m)) /. zm))
    xs


let rec differentiate z =
  match z.gradient with
  | Some(g) -> g
  | None ->
    let g = List.fold_right z.descendents  ~init:0.0 ~f:(fun (d,partial) a -> a +. differentiate d *. partial) in
    z.gradient <- Some(g);
    g

let rec zero_gradients z =
  match z.gradient with
  | None -> ()
  | Some(_) -> begin
      List.iter z.arguments ~f:(fun a -> zero_gradients a);
      z.gradient <- None;
      z.descendents <- [];
      if z.arguments = [] then () else z.data <- None
    end

let rec forward z =
  match z.data with
  | Some(v) -> v
  | None ->
    let argument_values = List.map z.arguments ~f:forward in
    let v = z.forwardProcedure argument_values in
    let partials = z.backwardProcedure argument_values in
    List.iter2_exn partials z.arguments ~f:(fun p a ->
        a.descendents <- (z,p)::a.descendents);
    z.data <- Some(v);
    v

let backward z =
  z.gradient <- Some(1.0);
  let rec b v =
    ignore(differentiate v);
    List.iter ~f:b v.arguments
  in b z

let update_network loss =
  zero_gradients loss;
  let l = forward loss in
  backward loss;
  l

let rec gradient_descent ?update:(update = 1000)
    ?iterations:(iterations = 10000) ?lr:(lr = 0.001) loss parameters =
  if iterations = 0 then parameters else begin 
    let l = update_network loss in
    if iterations mod update = 0 then begin 
      Printf.printf "LOSS: %f\n" l;
      parameters |> List.iter ~f:(fun p -> Printf.printf "parameter %f\t" (p.data |> get_some));
      Printf.printf "\n";
    end else ();
    let gradient = parameters |> List.map ~f:differentiate in
    List.iter2_exn parameters gradient ~f:(fun x dx ->
        let v = x.data |> get_some in
        update_variable x (v -. dx*.lr));
    gradient_descent ~iterations:(iterations - 1) ~lr:lr loss parameters
  end  
  
  

let test_differentiation () =
  let x = ~$ 10.0 in
  let y = ~$ 2. in
  let z = x -& log_soft_max [x;y;] in
  backward z;
  Printf.printf "dL/dx = %f\tdL/dy = %f\n" (differentiate x) (differentiate y);

  update_variable x 2.;
  update_variable y 10.;

  ignore(update_network z);

  Printf.printf "dL/dx = %f\tdL/dy = %f\n" (differentiate x) (differentiate y);  

  update_variable x 2.;
  update_variable y 2.;

  ignore(update_network z);

  Printf.printf "z = %f\n" (z.data |> get_some);

  Printf.printf "dL/dx = %f\tdL/dy = %f\n" (differentiate x) (differentiate y);

  let l = ((~$ 0.) -& z) in
  ignore(gradient_descent l [x;y])

;;

test_differentiation();;
