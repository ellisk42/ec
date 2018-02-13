open Core.Std

type 'a parsing = (string -> ('a*string) list)

let return_parse (x : 'a) : 'a parsing =
  fun s -> [(x,s)]

let parse_failure : 'a parsing =
  fun s -> []

let bind_parse (x : 'a parsing) (f : 'a -> 'b parsing) : 'b parsing =
  fun s ->
    x s |> List.map ~f:(fun (xp,suffix) -> f xp suffix) |> List.concat

let (%%) = bind_parse

let (<|>) (x : 'a parsing) (y : 'a parsing) : 'a parsing =
  fun s ->
    x s @ y s

let string_take_while f s =
  let rec check j =
    if j < String.length s && f s.[j] then check (j+1) else j
  in String.prefix s (check 0)

let constant_parser k : unit parsing =
  fun x -> if String.is_prefix x ~prefix:k then [((), String.drop_prefix x (String.length k))] else []

let token_parser ?can_be_empty:(can_be_empty = false) (element : char -> bool) : string parsing =
  fun x ->
    let token = string_take_while element x in
    if (not can_be_empty) && String.length token = 0 then [] else
      [(token, String.drop_prefix x (String.length token))]

let run_parser (p : 'a parsing) (s : string) : 'a option =
  p s |> List.fold_right ~init:None ~f:(fun (r,suffix) a ->
      match a with
      | Some(_) -> a
      | None ->
        if String.length suffix = 0 then Some(r) else None)
