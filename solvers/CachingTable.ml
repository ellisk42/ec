open Core

module CachingTable = struct
  type 'a node =
    { node_key : 'a;
      mutable next : 'a node option;
      mutable previous : 'a node option;}

  type ('a, 'b) t =
    { mutable oldest_key : 'a node option;
      mutable newest_key : 'a node option;
      mapping : ('a, ('b*'a node)) Hashtbl.t;
      capacity : int;
    }

  let length m = Hashtbl.length m.mapping

  let create capacity =
    assert (capacity > 3);
    {oldest_key = None;
     newest_key = None;
     mapping = Hashtbl.Poly.create();
     capacity;}

  let refresh m n =
    match m.newest_key with
    | Some(newest) when newest == n -> ()
    | _ ->

      (* Remove n from doubly linked list *)
      (match n.previous with
       | None ->
         (match m.oldest_key with
          | Some(n') ->
            assert (n == n');
            m.oldest_key <- n.next
          | None -> assert (false))
       | Some(p) -> p.next <- n.next);
      (match n.next with
       | None -> assert (false) (* this would mean that we are the most recent *)
       | Some(successor) -> successor.previous <- n.previous);

      (match m.oldest_key with
       | Some(o) when o == n -> assert (false)
       | None | Some(_) -> ());
      (match m.newest_key with
       | Some(newest) when newest == n -> assert (false)
       | None | Some(_) -> ());

      (* insert at the front of list *)
      n.previous <- m.newest_key;
      n.next <- None;
      (match m.newest_key with
       | None -> ()
       | Some(old_newest) -> old_newest.next <- Some(n));
      m.newest_key <- Some(n)

  let collect m =
    if Hashtbl.length m.mapping <= m.capacity then () else
      match m.oldest_key with
      | None -> assert (false)
      | Some(entry) ->
        Hashtbl.remove m.mapping entry.node_key;
        m.oldest_key <- entry.next

  let historical m =
    let rec forward = function
      | None -> []
      | Some(e) -> e.node_key :: forward e.next
    in forward m.oldest_key

  let backward_historical m =
    let rec backward = function
      | None -> []
      | Some(e) -> e.node_key :: backward e.previous
    in backward m.newest_key

  let find m k =
    match Hashtbl.find m.mapping k with
    | None -> None
    | Some((v,n)) ->
      refresh m n;
      Some(v)

  let set m k v =
    match Hashtbl.find m.mapping k with
    | None ->
      (* Create a new entry and put it at the front *)
      let entry = { node_key = k;
                    next = None;
                    previous = m.newest_key;} in
      (match m.newest_key with
       | Some(old_newest) -> old_newest.next <- Some(entry)
       | None -> ());
      m.newest_key <- Some(entry);
      (match m.oldest_key with
       | None -> m.oldest_key <- Some(entry)
       | Some(_) -> ());
      assert (Hashtbl.add m.mapping ~key:k ~data:(v, entry) = `Ok);
      collect m

    | Some((_,entry)) ->
      Hashtbl.set m.mapping ~key:k ~data:(v, entry);
      refresh m entry

  let check_consistency m =
    let rec forward e =
      match e.next with
      | Some(successor) ->
        (match successor.previous with
         | None -> assert (false)
         | Some(this) ->
           assert (this == e);
           forward successor)
      | None ->
        match m.newest_key with
        | None -> assert (false)
        | Some(this) ->
          assert (this.node_key == e.node_key)

    in

    let rec backward e =
      match e.previous with
      | Some(predecessor) ->
        (match predecessor.next with
         | None -> assert (false)
         | Some(this) ->
           assert (this == e);
           backward predecessor)
      | None ->
        match m.oldest_key with
        | Some(this) -> assert (this == e)
        | None -> assert (false)
    in

    (match m.newest_key, m.oldest_key with
     | None, None -> ()
     | Some(newest), Some(oldest) ->
       (assert (oldest.previous = None);
        assert (newest.next = None);
        forward oldest;
        backward newest)
     | None, Some(_) -> assert (false)
     | Some(_), None -> assert (false));

    let rec list_mapping = function
      | None -> []
      | Some(e) -> e :: list_mapping e.next
    in

    let entries = list_mapping m.oldest_key in
    entries |> List.iter ~f:(fun entry ->
        match Hashtbl.find m.mapping entry.node_key with
        | None -> assert (false)
        | Some(_,entry') -> assert (entry == entry'));

    Hashtbl.iteri m.mapping ~f:(fun ~key ~data:(_,entry) ->
        assert (1 =
                (entries |> List.filter ~f:(fun entry' ->
                     if entry' == entry then
                       (assert (entry'.node_key == key);
                        true)
                     else false) |> List.length)))

  let test() =
    let capacity = 10 in
    let m = create capacity in
    let ground_truth = Hashtbl.Poly.create() in

    let step() =
      let k = Random.int 10 in
      let v = Random.int 10 in

      Printf.eprintf "t[%d] = %d\n" k v;
      set m k v;
      Hashtbl.set ground_truth ~key:k ~data:v;
      check_consistency m;
      match find m k with
      | None -> assert (false)
      | Some(v') -> assert (v = v'); assert (v = Hashtbl.find_exn ground_truth k);  check_consistency m
    in

    for i = 1 to 100 do
      step();

      historical m |> List.iter ~f:(Printf.eprintf "%d ");
      Printf.eprintf "\n";
      backward_historical m |> List.rev |> List.iter ~f:(Printf.eprintf "%d ");
      Printf.eprintf "\n"
    done  
    
end;;
  

(* CachingTable.test() *)
