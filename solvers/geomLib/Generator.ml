open Interpreter

exception InternalGenerationError of string

let dummy_var = Unit
let dummy_program = Turn(None)

let valuesCostVar v =
    let default = (1. /. (float_of_int (Interpreter.valuesCostVar v))) in
    match v with
     (*| Indefinite -> 0.1 *. default*)
     | _ -> default

let valuesCostProgram p =
    let default = (1. /. (float_of_int (Interpreter.valuesCostProgram p))) in
    match p with
    | Repeat(_,_) -> 3. *. default
    | Integrate(_,_,_) -> 3. *. default
    | Turn(_) -> 3. *. default
    | Concat(_,_) -> 2. *. default
    | _ -> default

let total_var_op   = 0.
    +. (valuesCostVar (Double Unit))
    +. (valuesCostVar (Half Unit))
    +. (valuesCostVar (Next Unit))
    +. (valuesCostVar (Prev Unit))
    +. (valuesCostVar (Opposite Unit))
    +. (valuesCostVar (Divide (Unit,Unit)))

let cumsum_op_htbl = Hashtbl.create 101

let cumsum_op var =
    let rec helper var = match var with
    | Double v -> (valuesCostVar (Double v))
    | Half v -> helper (Double v) +. (valuesCostVar (Half v))
    | Next v -> helper (Half v) +. (valuesCostVar (Next v))
    | Prev v -> helper (Next v) +. (valuesCostVar (Prev v))
    | Opposite v -> helper (Prev v) +. (valuesCostVar (Opposite v))
    | Divide(v1,v2) -> helper (Opposite v1) +. (valuesCostVar (Divide(v1,v2)))
    | _ -> raise (InternalGenerationError("in cumsum_op"))
    in if Hashtbl.mem cumsum_op_htbl var
    then Hashtbl.find cumsum_op_htbl var
    else
        let value = helper var in
        Hashtbl.add cumsum_op_htbl var value ;
        value

let total_var_unit = 0.
    +. (valuesCostVar Unit)
    (*+. (valuesCostVar Indefinite)*)
    +. (valuesCostVar (Name ""))

let cumsum_unit_htbl = Hashtbl.create 101

let cumsum_unit var =
    let rec helper var = match var with
    | Unit -> valuesCostVar Unit
    (*| Indefinite -> helper Unit +. (valuesCostVar Indefinite)*)
    | Name _ -> helper Unit +. (valuesCostVar (Name ""))
    | _ -> raise (InternalGenerationError("in cumsum_unit"))
    in if Hashtbl.mem cumsum_unit_htbl var
    then Hashtbl.find cumsum_unit_htbl var
    else
        let value = helper var in
        Hashtbl.add cumsum_unit_htbl var value ;
        value

let total_program = 0.
    +. (valuesCostProgram (Turn(None)))
    +. (valuesCostProgram (Embed(dummy_program)))
    +. (valuesCostProgram (Concat(dummy_program,dummy_program)))
    +. (valuesCostProgram (Repeat(None,dummy_program)))
    +. (valuesCostProgram (Integrate(None,None,(None,None,None,None))))

let cumsum_program_htbl = Hashtbl.create 101

let cumsum_program p =
    let rec helper p = match p with
    | Turn v -> valuesCostProgram (Turn(v))
    | Embed p -> (helper (Turn(None))) +. (valuesCostProgram (Embed(p)))
    | Concat (p1,p2) ->
        (helper (Embed(dummy_program))) +. (valuesCostProgram (Concat(p1,p2)))
    | Repeat (r,p) ->
        (helper (Concat(dummy_program,dummy_program)))
        +. (valuesCostProgram (Repeat(r,p)))
    | Define (s,v) ->
        (helper (Repeat(None,dummy_program)))
        +. (valuesCostProgram (Define(s,v)))
    | Integrate (v1,v2,v3) ->
        (helper (Define("",dummy_var)))
        +. (valuesCostProgram (Integrate(v1,v2,v3)))
    | Nop -> 0.
    in if Hashtbl.mem cumsum_program_htbl p
    then Hashtbl.find cumsum_program_htbl p
    else
        let value = helper p in
        Hashtbl.add cumsum_program_htbl p value ;
        value

let pick_random_in_list l =
    let length = List.length l in
    let ith = Random.int length in
    List.nth l ith

let rec get_random_var : string list -> var = fun var_list ->
    if Random.bool () then
        match Random.float total_var_unit with
        | n when n < cumsum_unit Unit -> Unit
        (*| n when n < cumsum_unit Indefinite -> Indefinite*)
        | n when n < cumsum_unit (Name "") ->
            begin
                match var_list with 
                | [] -> get_random_var var_list
                | _ -> Name(pick_random_in_list var_list)
            end
        | n ->
            raise (InternalGenerationError("in total_var_unit"))
    else
        match Random.float total_var_op with
        | n when n < cumsum_op (Double(dummy_var)) ->
            Double (get_random_var var_list)
        | n when n < cumsum_op (Half(dummy_var)) ->
            Half (get_random_var var_list)
        | n when n < cumsum_op (Next(dummy_var)) ->
            Next (get_random_var var_list)
        | n when n < cumsum_op (Prev(dummy_var)) ->
            Prev (get_random_var var_list)
        | n when n < cumsum_op (Opposite(dummy_var)) ->
            Opposite (get_random_var var_list)
        | n when n < cumsum_op (Divide(dummy_var,dummy_var)) ->
            Divide (get_random_var var_list, get_random_var var_list)
        | _ -> raise (InternalGenerationError("in total_var_op"))


let rec_generate_random : string list -> (string list * shapeprogram) =
    fun var_list ->
    let nonEmpty = ref false in
    let rec helper = fun var_list ->
    match Random.float total_program with
    | n when n < cumsum_program (Turn(None)) ->
        let b = Random.bool () in
        let var = if b then Some(get_random_var var_list) else None in
        (var_list,Turn(var))
    | n when n < cumsum_program (Embed(dummy_program)) ->
        let l,p = helper var_list in
        (var_list,Embed(p))
    | n when n < cumsum_program (Concat(dummy_program,dummy_program)) ->
        let l,p = helper var_list in
        let l',p' = helper l in
        (l',Concat(p,p'))
    | n when n < cumsum_program (Repeat(None,dummy_program)) ->
        let b = Random.bool () in
        let var = if b then Some(get_random_var var_list) else None in
        let l,p = helper var_list in
        l,Repeat(var, p)
    | n when n < cumsum_program (Define("",dummy_var)) ->
        let var = get_random_var var_list in
        let length = List.length var_list in
        let new_var_name = Printf.sprintf "v%d" (length+1) in
        ((new_var_name::var_list),Define(new_var_name,var))
    | n when n < cumsum_program (Integrate(None,None,(None,None,None,None))) ->
        let varArray = Array.make 5 None in
        for i = 0 to 4 do
            if Random.int 10 = 0 then
            varArray.(i) <- Some(get_random_var var_list)
        done ;
        let pen = if Random.bool () then None else Some(Random.bool ()) in
        if pen = None || pen = Some(true) then nonEmpty := true ;
        (var_list,
         Integrate(varArray.(0),
                  (pen),
                  (varArray.(1),
                   varArray.(2),
                   varArray.(3),
                   None)))
    | _ -> raise (InternalGenerationError("in rec_generate_random"))
    in let l = ref [] and p = ref (Turn(None)) in
    while not (!nonEmpty) do
        let (ll,pp) = helper var_list in
        if !nonEmpty then (l := ll ; p := pp)
    done;
    (!l,!p)

let generate_random : unit -> shapeprogram =
    fun () ->
        let (_,p) = match Random.int 10 with
        | n when n < 1 -> rec_generate_random []
        | n when n < 3 ->
            let p1 = Define("v1", get_random_var []) in
            let (_,p2) = rec_generate_random ["v1"] in
            ([],Concat(p1,p2))
        | n when n < 6 ->
            let (l,p1) = rec_generate_random [] in
            let (_,p2) = rec_generate_random l in
            ([],Concat(p1,p2))
        | _ ->
            let var = get_random_var [] in
            let l,p = rec_generate_random [] in
            ([], Repeat(Some var,p))
        in p
