(*
  funarray.mli
   
  Port of Chris Okasaki's purely functional 
  random-access list to CAML: supports random access
  and pure functional updates AND supports head/tail 
  operations in O(1)
  
  Construct a random-access list with 
    cons elt empty
  -> returns elt::empty

  Access an element of a random-access list with
    lookup ls idx
  -> returns element at idx

  Update an element of a random-access list with
    update ls idx new
  -> returns new list with new replacing former element at idx

---

  ported by Will Benton, 10/5/2004 

  distributed under the GNU GPL
*)

type 'a funarray (* functional array type *)

exception Subscript
exception Empty

val empty : 'a funarray
val lookup : 'a funarray -> int -> 'a
val update : 'a funarray -> int -> 'a -> 'a funarray
val isempty : 'a funarray -> bool
val cons : 'a -> 'a funarray -> 'a funarray
val head : 'a funarray -> 'a
val tail : 'a funarray -> 'a funarray
