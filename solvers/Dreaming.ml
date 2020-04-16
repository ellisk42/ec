open Core

open Pregex
open Program
open Enumeration
open Grammar
open Utils
open Timeout
open Type
open Tower
open PolyValue

open Yojson.Basic



let remove_bad_dreams behavior_to_programs : (PolyList.t * (float * program list)) list =
  let start_time = Time.now () in

  (* remove the extra costs *)
  let behavior_to_programs =
    let btp = make_poly_list_table() in
    Hashtbl.iteri behavior_to_programs ~f:(fun ~key ~data ->
        let l,_,ps = data in
        Hashtbl.set btp ~key:key ~data:(l,ps));
    btp
  in

  (* number of inputs *)
  let l = ref None in
  Hashtbl.iteri behavior_to_programs ~f:(fun ~key ~data ->
      match !l with
      | None -> l := Some(List.length key)
      | Some(l') -> assert (List.length key = l'));
  let l = !l |> get_some in
  
  let containers = Array.init l  ~f:(fun _ -> make_poly_table()) in
  let output_vectors = empty_resizable() in
  
  Hashtbl.iteri behavior_to_programs ~f:(fun ~key ~data ->
      let this_index = output_vectors.ra_occupancy in
      push_resizable output_vectors (key, data);

      let outputs = key in
      outputs |> List.iteri ~f:(fun output_index this_output ->
          (* Record that we are one of the behaviors that produces this output *)
          if this_output = PolyValue.None then () else
            match Hashtbl.find containers.(output_index) this_output with
            | None -> Hashtbl.set containers.(output_index) ~key:this_output
                        ~data:(Int.Set.singleton this_index)
            | Some(others) -> Hashtbl.set containers.(output_index) ~key:this_output
                                ~data:(Int.Set.add others this_index)
        ));

  (* Checks whether there exists another output vector that contains everything in this vector *)
  let is_bad_index i =
    let dominating = ref None in  
    let outputs, _ = get_resizable output_vectors i in
    (* Initialize dominating to be the smallest set *)
    outputs |> List.iteri ~f:(fun output_index this_output ->
        if this_output = PolyValue.None then () else
          match Hashtbl.find containers.(output_index) this_output with
          | None -> assert (false)
          | Some(others) ->
            match !dominating with
            | Some(d) when Int.Set.length d > Int.Set.length others -> dominating := Some(others)
            | _ -> ());

    outputs |> List.iteri ~f:(fun output_index this_output ->
        if this_output = PolyValue.None then () else
          match Hashtbl.find containers.(output_index) this_output with
          | None -> assert (false)
          | Some(others) ->
            match !dominating with
            | None -> dominating := Some(others)
            | Some(d) -> dominating := Some(Int.Set.inter d others));
    let nightmare = Int.Set.length (!dominating |> get_some) > 1 in
    if nightmare && false then begin 
      Printf.eprintf "NIGHTMARE!!!";
      get_resizable output_vectors i |> snd |> snd |> List.iter ~f:(fun p -> p |> string_of_program |> Printf.eprintf "%s\n")
      (* get_resizable output_vectors i |> fst |> List.iter2_exn extra ~f:(fun i pv -> *)
      (*     Printf.eprintf "%s -> %s\n" (PolyValue.to_string i) (PolyValue.to_string pv)) *)
    end;
    nightmare
  in

  let number_of_nightmares = ref 0 in
  let sweet_dreams = 
    List.range 0 output_vectors.ra_occupancy |>
    List.filter_map ~f:(fun i ->
        if is_bad_index i then (incr number_of_nightmares; None) else
          Some(get_resizable output_vectors i))  
  in
  Printf.eprintf "Removed %d nightmares in %s.\n"
    (!number_of_nightmares) (Time.diff (Time.now ()) start_time |> Time.Span.to_string);
  sweet_dreams

  
let helmholtz_enumeration (behavior_hash : program -> (PolyList.t*float) option) ?nc:(nc=1) g request ~timeout ~maximumSize =
  assert (nc = 1); (* FIXME *)

  (* map from behavior signature to (nll of programs with signature, minimal auxiliary cost, list of MDL programs with signature) *)
  let behavior_to_programs = make_poly_list_table() in

  let update ~key ~data =
    let l,extra_cost,ps = data in
    match Hashtbl.find behavior_to_programs key with
    (* never seen this behavior before *)
    | None -> Hashtbl.set behavior_to_programs ~key ~data:data
    (* have seen this behavior before and our cost is strictly better, OR our cost is the same but we are more likely *)
    | Some((l',extra_cost',_))
      when extra_cost < extra_cost' || (extra_cost = extra_cost' && l > l')
      -> Hashtbl.set behavior_to_programs ~key ~data
    (* have seen this behavior before and our cost is strictly worst, OR our cost is the same but we are less likely *)
    | Some((l',extra_cost',_))
      when extra_cost > extra_cost' || (extra_cost = extra_cost' && l' < l)
      -> ()
    (* we are the same cost but less likely *)
    | Some((l',extra_cost',ps'))
      when extra_cost = extra_cost' && l < l'
      -> ()
    (* we cannot be a different cost or of the other conditions would've fired *)
    (* so we're the same cost, but also we know that we are not less likely *)
    (* also we cannot be more likely or the first condition would have held *)
    | Some((l',extra_cost',ps')) ->
      (* If our extra costs were different, than either of the above conditions would have triggered *)
      (* therefore we have the same extra cost *)
      assert (extra_cost = extra_cost');
      (* Given that the above holds, we also know that l' = l *)
      if not (l = l') then Printf.eprintf "costs:\t%f\t%flikelihood:\t%f\t%f\n"
          extra_cost extra_cost' l l';
      assert (l = l');
      Hashtbl.set behavior_to_programs ~key ~data:(l, extra_cost, ps @ ps' |> List.dedup_and_sort ~compare:compare_program)
  in

  (* unused *)
  let merge other =
    Hashtbl.iteri other ~f:update
  in 

  set_enumeration_timeout timeout;

  let rec loop lb =
    if enumeration_timed_out() then () else begin 
      let final_results = 
        enumerate_programs ~extraQuiet:true ~nc:nc ~final:(fun () -> [behavior_to_programs])
          g request lb (lb+.1.5) (fun p l ->
              if Hashtbl.length behavior_to_programs > maximumSize then set_enumeration_timeout (-1.0) else
                match behavior_hash p with
                | Some(key, extra_cost) -> update ~key ~data:(l,extra_cost,[p])
                | None -> ()
            ) |> List.concat
      in
      if nc > 1 then final_results |> List.iter ~f:merge;
      loop (lb+.1.5)
    end
  in

  loop 0.;

  behavior_to_programs

let rec unpack x =
  let open Yojson.Basic.Util in
  
  try magical (x |> to_int) with _ ->
  try magical (x |> to_float) with _ ->
  try magical (x |> to_bool) with _ ->
  try
    let v = x |> to_string in
    if String.length v = 1 then magical v.[0] else magical v
  with _ ->
  try
    x |> to_list |> List.map ~f:unpack |> magical
  with _ -> raise (Failure "could not unpack")

let rec pack t v : json =
  let open Yojson.Basic.Util in
  match t with
  | TCon("list",[t'],_) -> `List(magical v |> List.map ~f:(pack t'))
  | TCon("int",[],_) -> `Int(magical v)
  | TCon("bool",[],_) -> `Bool(magical v)
  | TCon("char",[],_) -> `String(magical v |> String.of_char)
  | _ -> assert false

let special_helmholtz =   Hashtbl.Poly.create();;
let register_special_helmholtz name handle = Hashtbl.set special_helmholtz name handle;;


let default_hash ?timeout:(timeout=0.001) request inputs : program -> (PolyList.t*float) option =
  let open Yojson.Basic.Util in

  (* convert json -> ocaml *)
  let inputs : 'a list list = unpack inputs in
  let return = return_of_type request in

  fun program ->
    let p = analyze_lazy_evaluation program in
    let outputs = inputs |> List.map ~f:(fun input ->
        try
          match run_for_interval ~attempts:2 timeout
                  (fun () -> run_lazy_analyzed_with_arguments p input)
          with
          | Some(value) -> PolyValue.pack return value            
          | _ -> PolyValue.None
        with (* We have to be a bit careful with exceptions if the
              * synthesized program generated an exception, then we just
              * terminate w/ false but if the enumeration timeout was
              * triggered during program evaluation, we need to pass the
              * exception on
             *)
        | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
        | _                   -> PolyValue.None) in
    if List.exists outputs ~f:PolyValue.is_some then
      Some(outputs,0.)
    else None

let string_hash ?timeout:(timeout=0.001) request inputs : program -> (PolyList.t*float) option =
  let open Yojson.Basic.Util in

  (* convert json -> ocaml *)
  let inputs : 'a list list = unpack inputs in
  let return = return_of_type request in

  let testConstants=["x4";"a bc d"]  in
  let constants = testConstants |> List.map ~f:String.to_list in 

  fun program ->
    let constant_results = (* results from substituting with each constant *)
      constants |> List.concat_map ~f:(fun constant ->
          match substitute_string_constants [constant] program with
          | [program'] -> 
            let p = analyze_lazy_evaluation program' in    
            inputs |> List.map ~f:(fun input ->
                try
                  match run_for_interval ~attempts:2
                          timeout (fun () -> run_lazy_analyzed_with_arguments p input)
                  with
                  | Some(value) -> PolyValue.pack return value
                  | _ -> PolyValue.None
                with
                | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
                | _                   -> PolyValue.None)
          | _ -> assert false)
    in
    if List.exists constant_results ~f:PolyValue.is_some then
      Some(constant_results,0.)
    else None
;;

register_special_helmholtz "string" string_hash;;



(* let rational_hash ?timeout:(timeout=0.001) request inputs : program -> PolyList.t option = *)
(*   assert (request = (treal @> treal)); *)

(*   let open Differentiation in *)

(*   let number_of_constant_sequences = 1 in *)
(*   let max_parameters = 5 in *)
(*   let constant_sequences = List.range 0 number_of_constant_sequences |> *)
(*                            List.map ~f:(fun _ -> *)
(*                                let c = Random.float_range (-3.) (3.) in *)
(*                                List.range 0 max_parameters |> List.map ~f:(fun _ -> c)) *)
(*   in *)

(*   let test_inputs = (0--30) |> List.map ~f:(fun _ -> Random.float_range (-10.) (10.)) in *)

(*   let rec substitute program constant_sequence = match program with *)
(*     | Primitive(t,"REAL",_) -> begin *)
(*         let v = random_variable() in *)
(*         update_variable v (List.hd_exn constant_sequence); *)
(*         Primitive(t,"REAL", ref v |> magical), List.tl_exn constant_sequence *)
(*       end *)
(*     | Invented(_,b) -> substitute b constant_sequence *)
(*     | Abstraction(b) -> *)
(*       let b',s' = substitute b constant_sequence in *)
(*       Abstraction(b'), s' *)
(*     | Apply(f,x) -> *)
(*       let f',s' = substitute f constant_sequence in *)
(*       let x',s'' = substitute x constant_sequence in *)
(*       Apply(f',x'), s'' *)
(*     | Index(_) | Primitive(_,_,_) -> program, constant_sequence *)
(*   in  *)

(*   fun program -> *)
(*     constant_sequences |> List.map ~f:(fun s -> *)
(*         test_inputs |> List.map ~f:(fun x -> *)
(*             let x' = placeholder_data treal x in *)
(*             let p = substitute program s in *)
(*             match  *)
(*               try *)
(*                 run_for_internal ~attempts:2 timeout *)
(*                   (fun () -> run_lazy_analyzed_with_arguments (analyze_lazy_evaluation p) [x']) *)
(*               with _ -> None *)
(*             with *)
(*             | None -> None *)
(*             |  *)
  
  

let tower_hash ?timeout:(timeout=0.001) request inputs : program -> (PolyList.t*float) option =
  let open Yojson.Basic.Util in

  assert (request = (ttower @> ttower));
  
  fun program ->
    let arrangement = evaluate_discrete_tower_program timeout program in
    let l = List.length arrangement in
    let w = blocks_extent arrangement in
    let h = tower_height arrangement in
    if l = 0 || l > 100 || w > 360 || h > 250 then None else
      let j = PolyValue.List(arrangement |> List.map ~f:(fun (a,b,c,d) ->
          PolyValue.List([PolyValue.Integer(a);
                          PolyValue.Integer(b);
                          PolyValue.Integer(c);
                          PolyValue.Integer(d);]))) in
      Some([j],0.)
;;
register_special_helmholtz "tower" tower_hash;;

let logo_hash ?timeout:(timeout=0.001) request inputs : program -> (PolyList.t*float) option =
  let open Yojson.Basic.Util in

  assert (request = (turtle @> turtle));
  (* disgusting hack *)
  let costMatters = 1 = (inputs |> to_list |> List.hd_exn |> to_list |> List.hd_exn |> to_int) in
  
  let table = Hashtbl.Poly.create() in

  fun program ->
    let p = analyze_lazy_evaluation program in
    let l = run_for_interval ~attempts:2 timeout (fun () ->
        let x = run_lazy_analyzed_with_arguments p [] in
        let l = LogoLib.LogoInterpreter.turtle_to_list x in
        if not (LogoLib.LogoInterpreter.logo_contained_in_canvas l) then None else 
          match Hashtbl.find table l with
          | Some(a) -> Some(a)
          | None -> begin
              let a = LogoLib.LogoInterpreter.turtle_to_array_and_cost x 28 in
              Hashtbl.set table ~key:l ~data:a;
              Some(a)
            end)
    in
    match l with
    | None -> None (* timeout *)
    | Some(None) -> None (* escaped the canvas *)
    | Some(Some(a,cost)) ->
      let j = PolyValue.List(range (28*28) |> List.map ~f:(fun i -> PolyValue.Integer(a.{i}))) in
      let cost = if costMatters then cost else 0. in
      Some([j],cost);;
register_special_helmholtz "LOGO" logo_hash;;

let regex_hash  ?timeout:(timeout=0.001) request inputs : program -> (PolyList.t*float) option =
  let open Yojson.Basic.Util in
  assert (request = (tregex @> tregex));

  let rec poly_of_regex = function
    | Constant(s) -> PolyValue.List ([PolyValue.Integer(0);
                                      poly_of_string s])
    | Kleene(p) -> PolyValue.List([PolyValue.Integer(1); poly_of_regex p])
    | Plus(p) -> PolyValue.List([PolyValue.Integer(2); poly_of_regex p])
    | Maybe(p) -> PolyValue.List([PolyValue.Integer(3); poly_of_regex p])
    | Alt(p,q) -> PolyValue.List([PolyValue.Integer(4); poly_of_regex p; poly_of_regex q])
    | Concat(p,q) -> PolyValue.List([PolyValue.Integer(5); poly_of_regex p; poly_of_regex q])
  and poly_of_string = function
    | String(s) -> PolyValue.List(List.map ~f:(fun c -> PolyValue.Character(c)) s)
    | Dot -> PolyValue.Integer(0)
    | D -> PolyValue.Integer(1)
    | S -> PolyValue.Integer(2)
    | W -> PolyValue.Integer(3)
    | L -> PolyValue.Integer(4)
    | U -> PolyValue.Integer(5)
  in
  let default_constant = build_constant_regex ['c';'o';'n';'s';'t';'9';'#';] in
  fun expression ->
    if number_of_free_parameters expression > 1 then None else 
      run_for_interval ~attempts:2 timeout
        (fun () -> 
           let r = expression |> substitute_constant_regex default_constant |>
                   regex_of_program |> canonical_regex in
           ([poly_of_regex r],0.))
;;
register_special_helmholtz "regex" regex_hash;;
