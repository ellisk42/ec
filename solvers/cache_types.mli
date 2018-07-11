(** cache.proto Types *)



(** {2 Types} *)

type tower_cash_block = {
  x10 : int32;
  w10 : int32;
  h10 : int32;
}

type tower_cash_entry = {
  plan : tower_cash_block list;
  height : float;
  stability : float;
  area : float;
  length : float;
  overpass : float;
  staircase : float;
}

type tower_cash = {
  entries : tower_cash_entry list;
} [@@unboxed]


(** {2 Default values} *)

val default_tower_cash_block : 
  ?x10:int32 ->
  ?w10:int32 ->
  ?h10:int32 ->
  unit ->
  tower_cash_block
(** [default_tower_cash_block ()] is the default value for type [tower_cash_block] *)

val default_tower_cash_entry : 
  ?plan:tower_cash_block list ->
  ?height:float ->
  ?stability:float ->
  ?area:float ->
  ?length:float ->
  ?overpass:float ->
  ?staircase:float ->
  unit ->
  tower_cash_entry
(** [default_tower_cash_entry ()] is the default value for type [tower_cash_entry] *)

val default_tower_cash : 
  ?entries:tower_cash_entry list ->
  unit ->
  tower_cash
(** [default_tower_cash ()] is the default value for type [tower_cash] *)
