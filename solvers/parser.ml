open Core

type 'a parsing = (string*int -> ('a*int) list)

let return_parse (x : 'a) : 'a parsing =
  fun (s,n) -> [(x,n)]

let parse_failure : 'a parsing =
  fun (s,n) -> []

let bind_parse (x : 'a parsing) (f : 'a -> 'b parsing) : 'b parsing =
  fun (s,n) ->
    x (s,n) |> List.map ~f:(fun (xp,n') -> f xp (s,n')) |> List.concat

let (%%) = bind_parse

let (<|>) (x : 'a parsing) (y : 'a parsing) : 'a parsing =
  fun s ->
    x s @ y s

let constant_parser (k : string) : unit parsing =
  fun (s,n) ->
    let rec check consumed =
      if consumed = String.length k then true else
      if n + consumed >= String.length s || s.[n + consumed] <> k.[consumed] then false else
        check (consumed + 1)
    in
    if check 0 then [(),n + String.length k] else []

let token_parser ?can_be_empty:(can_be_empty = false) (element : char -> bool) : string parsing =
  fun (s,n) ->
    let rec check consumed =
      if n + consumed >= String.length s || (not (element s.[n + consumed])) then [] else
        s.[n + consumed] :: check (consumed + 1)
    in 
    let token = check 0 in
    if (not can_be_empty) && List.length token = 0 then [] else
      let token = String.concat ~sep:"" (token |> List.map ~f:(String.make 1))  in
      [(token, n + String.length token)]

let run_parser (p : 'a parsing) (s : string) : 'a option =
  p (s,0) |> List.fold_right ~init:None ~f:(fun (r,n) a ->
      match a with
      | Some(_) -> a
      | None ->
        if String.length s = n then Some(r) else None)
