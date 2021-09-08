open Core

open Utils
open Type


module PolyValue = struct
  type t =
    | List of t list
    | Integer of int
    | Float of float
    | Boolean of bool
    | Character of char
    | None
  [@@deriving compare, hash, sexp_of, equal]

  let rec pack t v : t =
    match t with
    | TCon("list",[t'],_) -> List(magical v |> List.map ~f:(pack t'))
    | TCon("int",[],_) -> Integer(magical v)
    | TCon("bool",[],_) -> Boolean(magical v)
    | TCon("char",[],_) -> Character(magical v)
    | _ -> assert false

  let is_some = function
    | None -> false
    | _ -> true

  let rec to_string = function
    | List(l) -> l |> List.map ~f:to_string |> join ~separator:";" |> Printf.sprintf "[%s]"
    | Integer(l) -> Printf.sprintf "%d" l
    | Float(f) -> Printf.sprintf "%f" f
    | Boolean(b) -> Printf.sprintf "%b" b
    | Character(c) -> Printf.sprintf "'%c'" c
    | None -> "None"

  let rec of_json (j : Yojson.Basic.t) : t = match j with
    | `List(l) -> List(l |> List.map ~f:of_json)
    | `Int(i) -> Integer(i)
    | `Bool(b) -> Boolean(b)
    | _ -> assert (false)

end;;

let make_poly_table() = Hashtbl.create (module PolyValue)


module PolyList = struct
  type t = PolyValue.t list
  [@@deriving compare, hash, sexp_of]
end;;

let make_poly_list_table() = Hashtbl.create (module PolyList)
