(** cache.proto Binary Encoding *)


(** {2 Protobuf Encoding} *)

val encode_tower_cash_block : Cache_types.tower_cash_block -> Pbrt.Encoder.t -> unit
(** [encode_tower_cash_block v encoder] encodes [v] with the given [encoder] *)

val encode_tower_cash_entry : Cache_types.tower_cash_entry -> Pbrt.Encoder.t -> unit
(** [encode_tower_cash_entry v encoder] encodes [v] with the given [encoder] *)

val encode_tower_cash : Cache_types.tower_cash -> Pbrt.Encoder.t -> unit
(** [encode_tower_cash v encoder] encodes [v] with the given [encoder] *)


(** {2 Protobuf Decoding} *)

val decode_tower_cash_block : Pbrt.Decoder.t -> Cache_types.tower_cash_block
(** [decode_tower_cash_block decoder] decodes a [tower_cash_block] value from [decoder] *)

val decode_tower_cash_entry : Pbrt.Decoder.t -> Cache_types.tower_cash_entry
(** [decode_tower_cash_entry decoder] decodes a [tower_cash_entry] value from [decoder] *)

val decode_tower_cash : Pbrt.Decoder.t -> Cache_types.tower_cash
(** [decode_tower_cash decoder] decodes a [tower_cash] value from [decoder] *)
