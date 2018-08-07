open Core

open Utils
open Program
    
(* equivalence class *)
type eq = {mutable leader : eq option;
           name : int;}
let compare_class k1 k2 =
  k1.name - k2.name

(* lifted expressions *)
type le =
  | TerminalLifted of program
  | ApplicationLifted of eq*eq
  | AbstractionLifted of eq

(* equivalence graph *)
type eg = {
  (* map from class to set of lexpressions *)
  members_of_class : (eq, le Hash_set.t) Hashtbl.t;
  (* map from le to class *)
  class_of_expression : (le, eq Hash_set.t) Hashtbl.t;
  (* map from class to all of the expressions incident on that class *)
  incident : (eq, le Hash_set.t) Hashtbl.t;
  (* if an expression belongs to more than one class than it is in this table *)
  number_of_classes : (le,int) Hashtbl.t;

  mutable next_class : int;
}

let rec chase e = match e.leader with
  | None -> e
  | Some(l) ->
    let l' = chase l in
    e.leader <- Some(l');
    l'

(* conveniently named helpers *)
let new_set() = Hash_set.Poly.create()
let member s x = Hash_set.mem s x
let insert s x = Hash_set.add s x
let remove s x = Hash_set.remove s x    
let set_size = Hash_set.length
let elements s = Hash_set.fold s ~f:(fun a x -> x :: a) ~init:[]
exception Dummy;;
let any_element s =
  let d = ref None in
  try
    Hash_set.iter s  ~f:(fun x ->
        d := Some(x);
        raise Dummy); assert false
  with Dummy -> !d |> get_some
let any_value t =
  let d = ref None in
  try
    Hashtbl.iteri t ~f:(fun ~key ~data:value ->
        d := Some(value);
        raise Dummy); assert false
  with Dummy -> !d |> get_some
let any_key t =                                           
  let d = ref None in
  try
    Hashtbl.iteri t ~f:(fun ~key ~data:value ->
        d := Some(key);
        raise Dummy); assert false
  with Dummy -> !d |> get_some

let new_class_graph() =
  {members_of_class = Hashtbl.Poly.create();
   class_of_expression = Hashtbl.Poly.create();
   incident = Hashtbl.Poly.create();
   number_of_classes = Hashtbl.Poly.create();
   next_class = 0;}

let new_class g =
  let k = {name = g.next_class; leader = None;} in
  g.next_class <- g.next_class + 1;
  Hashtbl.set g.members_of_class k (new_set());
  Hashtbl.set g.incident k (new_set());
  k

let add_edge g k l =
  insert (Hashtbl.find_exn g.members_of_class k) l;
  insert (Hashtbl.find_exn g.class_of_expression l) k;
  let sz = set_size (Hashtbl.find_exn g.class_of_expression l) in
  if sz > 1 then
    Hashtbl.set g.number_of_classes l sz
let delete_edge g k l =
  remove (Hashtbl.find_exn g.members_of_class k) l;
  remove (Hashtbl.find_exn g.class_of_expression l) k;
  match Hashtbl.find g.number_of_classes l with
  | None -> ()
  | Some(n) -> begin
      assert (n > 1);
      if n = 2 then
        Hashtbl.remove g.number_of_classes l
      else
        Hashtbl.set g.number_of_classes l (n - 1)
    end
let delete_class g k =
  assert (set_size (Hashtbl.find_exn g.members_of_class k) = 0);
  assert (set_size (Hashtbl.find_exn g.incident k) = 0);
  Hashtbl.remove g.members_of_class k;
  Hashtbl.remove g.incident k
let delete_expression g l =
  assert (set_size (Hashtbl.find_exn g.class_of_expression l) = 0);
  Hashtbl.remove g.class_of_expression l;
  match l with
  | AbstractionLifted(b) ->
    remove (Hashtbl.find_exn g.incident b) l
  | ApplicationLifted(f,x) ->
    remove (Hashtbl.find_exn g.incident f) l;
    remove (Hashtbl.find_exn g.incident x) l
  | _ -> ()
let rename g o n = (* renames expression oto n  *)
  Hashtbl.find_exn g.class_of_expression o |> Hash_set.to_list |> List.iter ~f:(fun refers ->
      delete_edge g refers o;
      add_edge g refers n);
  delete_expression g o

let incorporate_class g l = (* returns the class of an expression creating it if necessary *)
  if not (Hashtbl.mem g.class_of_expression l) then
    Hashtbl.set g.class_of_expression l (new_set());
  if set_size (Hashtbl.find_exn g.class_of_expression l) = 0 then
    let k = new_class g in
    add_edge g k l;
    k
  else
    Hashtbl.find_exn g.class_of_expression l |> any_element
let incorporate_expression g l = (* makes sure that it has a class and adds incident records *)
  match Hashtbl.find g.class_of_expression l with
  | Some(_) -> l (* already has a class *)
  | None ->
    Hashtbl.set g.class_of_expression l (new_set());
    (match l with
    | ApplicationLifted(f,x) ->
      (insert (Hashtbl.find_exn g.incident f) l;
       insert (Hashtbl.find_exn g.incident x) l)
    | AbstractionLifted(b) ->
      insert (Hashtbl.find_exn g.incident b) l
    | _ -> ());
    l

let apply_class g (f : eq) (x : eq) =
  incorporate_class g (incorporate_expression g @@ ApplicationLifted(chase f, chase x))
  
let abstract_class g (b : eq) =
  incorporate_class g (incorporate_expression g @@ AbstractionLifted(chase b))

let leaf_class g (l : program) =
  incorporate_class g (incorporate_expression g @@ TerminalLifted(l))

let set_leader g k2 k1 =
  let verify_dead k =
    let not_uses = function
      | TerminalLifted(_) -> ()
      | ApplicationLifted(f,x) -> begin
          assert (not (f = k));
          assert (not (x = k))
        end
      | AbstractionLifted(b) -> assert (not (b = k))
    in
    Hashtbl.iteri g.members_of_class ~f:(fun ~key ~data ->
        assert (not (key = k));
        data |> elements |> List.iter ~f:not_uses);
    Hashtbl.iteri g.class_of_expression ~f:(fun ~key ~data ->
        not_uses key;
        data |> elements |> List.iter ~f:(fun k' ->
            assert (not (k = k'))));
    Hashtbl.iteri g.incident ~f:(fun ~key ~data ->
        assert (not (key = k));
        data |> elements |> List.iter ~f:not_uses)    
  in 
  match k1.leader with
  | Some(_) -> assert false
  | None -> match k2.leader with
    | Some(_) -> assert false
    | None -> 
      ((* verify_dead k2; *)
       k2.leader <- Some(k1))
    

let rec make_equivalent g k1 k2 =
  let k1 = chase k1 in
  let k2 = chase k2 in
  if k1 = k2 then k1 else begin
    (* k2.leader <- Some(k1); *)
    Hashtbl.find_exn g.members_of_class k2 |> Hash_set.to_list |> List.iter ~f:(fun l ->
        add_edge g k1 l;
        delete_edge g k2 l);

    let update = function
      | ApplicationLifted(f,x) -> begin
          assert (f = k2 || x = k2);
          let f = if f = k2 then k1 else f in
          let x = if x = k2 then k1 else x in
          incorporate_expression g (ApplicationLifted(f,x))
        end
      | AbstractionLifted(b) -> begin
          assert (b = k2);
          incorporate_expression g (AbstractionLifted(b))
        end
      | _ -> assert false
    in

    Hashtbl.find_exn g.incident k2 |> Hash_set.to_list |> List.iter ~f:(fun l ->
        rename g l (update l));
    delete_class g k2;
    set_leader g k2 k1;
    merge g;
    k1
  end
and merge g =
  if Hashtbl.length g.number_of_classes > 0 then begin
    let l = any_key g.number_of_classes in
    let ks = Hashtbl.find_exn g.class_of_expression l in
    assert (set_size ks = Hashtbl.find_exn g.number_of_classes l);
    match elements ks with (* I hate core *)
    | k :: ks -> ks |> List.iter ~f:(fun k2 -> ignore(make_equivalent g k k2))
    | [] -> assert false    
  end
  
    

