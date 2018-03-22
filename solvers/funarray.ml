(*
  funarray.ml 
   
  Port of Chris Okasaki's purely functional 
  random-access list to CAML 
  
  Construct a random-access list with 
    cons elt empty
  
  Access an element of a random-access list with
    lookup ls idx

  Update an element of a random-access list with
    update ls idx new


  ported by Will Benton, 10/5/2004 

  distributed under the GNU GPL
*)

type 'a fatree =
    FALeaf of 'a
  | FANode of 'a * 'a fatree * 'a fatree

type 'a funarray = (int * 'a fatree) list

exception Subscript
exception Empty

let rec fatree_lookup size tree index =
  match (tree, index) with
      (FALeaf(x), 0) -> x
    | (FALeaf(x), i) -> raise Subscript
    | (FANode(x,t1,t2), 0) -> x
    | (FANode(x,t1,t2), i) ->
	let size' = size / 2 in
	  if i <= size' then 
	    fatree_lookup size' t1 (i - 1)
	  else
	    fatree_lookup size' t2 (i - 1 - size')

let rec fatree_update size tree index y =
  match (tree, index) with 
      (FALeaf(x), 0) -> FALeaf(y)
    | (FALeaf(x), i) -> raise Subscript
    | (FANode(x,t1,t2), 0) -> FANode(y,t1,t2)
    | (FANode(x,t1,t2), i) -> 
	 let size' = size / 2 in
	   if i <= size' then
	     FANode(x,fatree_update size' t1 (i - 1) y,t2)
	   else
	     FANode(x,t1,fatree_update size' t2 (i - 1 - size') y)

let rec lookup ls i =
    match (ls, i) with
	([], i) -> raise Subscript
      | ((size, t) :: rest, i) ->
	  if i < size then
	    fatree_lookup size t i
	  else
	    lookup rest (i - size)

let rec update ls i y =
  match (ls, i) with
      ([], i) -> raise Subscript
    | ((size, t) :: rest, i) ->
	if i < size then
	  (size, fatree_update size t i y) :: rest
	else
	  (size, t) :: update rest (i - size) y

let empty = []

let isempty ls =
  match ls with
      [] -> true
    | ((size,t) :: rest) -> false

let cons x ls =
  match (ls) with
      ((size1, t1) :: (size2, t2) :: rest) ->
	if size1 = size2 then
	  (1 + size1 + size2, FANode(x, t1, t2)) :: rest
	else
	  (1, FALeaf(x)) :: ls
    | xls -> (1, FALeaf(x)) :: xls

let head ls = 
  match ls with
      [] -> raise Empty
    | (size, FALeaf(x)) :: rest -> x
    | (size, FANode(x,t1,t2)) :: rest -> x
	
let tail ls =
  match ls with 
      [] -> raise Empty
    | (size, FALeaf(x)) :: rest -> rest
    | (size, FANode(x,t1,t2)) :: rest ->
	let size' = size / 2 in
	  (size', t1) :: (size', t2) :: rest

