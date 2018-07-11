[@@@ocaml.warning "-27-30-39"]

type tower_cash_block_mutable = {
  mutable x10 : int32;
  mutable w10 : int32;
  mutable h10 : int32;
}

let default_tower_cash_block_mutable () : tower_cash_block_mutable = {
  x10 = 0l;
  w10 = 0l;
  h10 = 0l;
}

type tower_cash_entry_mutable = {
  mutable plan : Cache_types.tower_cash_block list;
  mutable height : float;
  mutable stability : float;
  mutable area : float;
  mutable length : float;
  mutable overpass : float;
  mutable staircase : float;
}

let default_tower_cash_entry_mutable () : tower_cash_entry_mutable = {
  plan = [];
  height = 0.;
  stability = 0.;
  area = 0.;
  length = 0.;
  overpass = 0.;
  staircase = 0.;
}

type tower_cash_mutable = {
  mutable entries : Cache_types.tower_cash_entry list;
}

let default_tower_cash_mutable () : tower_cash_mutable = {
  entries = [];
}


let rec decode_tower_cash_block d =
  let v = default_tower_cash_block_mutable () in
  let continue__= ref true in
  let h10_is_set = ref false in
  let w10_is_set = ref false in
  let x10_is_set = ref false in
  while !continue__ do
    match Pbrt.Decoder.key d with
    | None -> (
    ); continue__ := false
    | Some (1, Pbrt.Varint) -> begin
      v.x10 <- Pbrt.Decoder.int32_as_varint d; x10_is_set := true;
    end
    | Some (1, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash_block), field(1)" pk
    | Some (2, Pbrt.Varint) -> begin
      v.w10 <- Pbrt.Decoder.int32_as_varint d; w10_is_set := true;
    end
    | Some (2, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash_block), field(2)" pk
    | Some (3, Pbrt.Varint) -> begin
      v.h10 <- Pbrt.Decoder.int32_as_varint d; h10_is_set := true;
    end
    | Some (3, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash_block), field(3)" pk
    | Some (_, payload_kind) -> Pbrt.Decoder.skip d payload_kind
  done;
  begin if not !h10_is_set then Pbrt.Decoder.missing_field "h10" end;
  begin if not !w10_is_set then Pbrt.Decoder.missing_field "w10" end;
  begin if not !x10_is_set then Pbrt.Decoder.missing_field "x10" end;
  ({
    Cache_types.x10 = v.x10;
    Cache_types.w10 = v.w10;
    Cache_types.h10 = v.h10;
  } : Cache_types.tower_cash_block)

let rec decode_tower_cash_entry d =
  let v = default_tower_cash_entry_mutable () in
  let continue__= ref true in
  let staircase_is_set = ref false in
  let overpass_is_set = ref false in
  let length_is_set = ref false in
  let area_is_set = ref false in
  let stability_is_set = ref false in
  let height_is_set = ref false in
  while !continue__ do
    match Pbrt.Decoder.key d with
    | None -> (
      v.plan <- List.rev v.plan;
    ); continue__ := false
    | Some (4, Pbrt.Bytes) -> begin
      v.plan <- (decode_tower_cash_block (Pbrt.Decoder.nested d)) :: v.plan;
    end
    | Some (4, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash_entry), field(4)" pk
    | Some (5, Pbrt.Bits32) -> begin
      v.height <- Pbrt.Decoder.float_as_bits32 d; height_is_set := true;
    end
    | Some (5, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash_entry), field(5)" pk
    | Some (6, Pbrt.Bits32) -> begin
      v.stability <- Pbrt.Decoder.float_as_bits32 d; stability_is_set := true;
    end
    | Some (6, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash_entry), field(6)" pk
    | Some (7, Pbrt.Bits32) -> begin
      v.area <- Pbrt.Decoder.float_as_bits32 d; area_is_set := true;
    end
    | Some (7, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash_entry), field(7)" pk
    | Some (8, Pbrt.Bits32) -> begin
      v.length <- Pbrt.Decoder.float_as_bits32 d; length_is_set := true;
    end
    | Some (8, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash_entry), field(8)" pk
    | Some (9, Pbrt.Bits32) -> begin
      v.overpass <- Pbrt.Decoder.float_as_bits32 d; overpass_is_set := true;
    end
    | Some (9, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash_entry), field(9)" pk
    | Some (10, Pbrt.Bits32) -> begin
      v.staircase <- Pbrt.Decoder.float_as_bits32 d; staircase_is_set := true;
    end
    | Some (10, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash_entry), field(10)" pk
    | Some (_, payload_kind) -> Pbrt.Decoder.skip d payload_kind
  done;
  begin if not !staircase_is_set then Pbrt.Decoder.missing_field "staircase" end;
  begin if not !overpass_is_set then Pbrt.Decoder.missing_field "overpass" end;
  begin if not !length_is_set then Pbrt.Decoder.missing_field "length" end;
  begin if not !area_is_set then Pbrt.Decoder.missing_field "area" end;
  begin if not !stability_is_set then Pbrt.Decoder.missing_field "stability" end;
  begin if not !height_is_set then Pbrt.Decoder.missing_field "height" end;
  ({
    Cache_types.plan = v.plan;
    Cache_types.height = v.height;
    Cache_types.stability = v.stability;
    Cache_types.area = v.area;
    Cache_types.length = v.length;
    Cache_types.overpass = v.overpass;
    Cache_types.staircase = v.staircase;
  } : Cache_types.tower_cash_entry)

let rec decode_tower_cash d =
  let v = default_tower_cash_mutable () in
  let continue__= ref true in
  while !continue__ do
    match Pbrt.Decoder.key d with
    | None -> (
      v.entries <- List.rev v.entries;
    ); continue__ := false
    | Some (1, Pbrt.Bytes) -> begin
      v.entries <- (decode_tower_cash_entry (Pbrt.Decoder.nested d)) :: v.entries;
    end
    | Some (1, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(tower_cash), field(1)" pk
    | Some (_, payload_kind) -> Pbrt.Decoder.skip d payload_kind
  done;
  ({
    Cache_types.entries = v.entries;
  } : Cache_types.tower_cash)

let rec encode_tower_cash_block (v:Cache_types.tower_cash_block) encoder = 
  Pbrt.Encoder.key (1, Pbrt.Varint) encoder; 
  Pbrt.Encoder.int32_as_varint v.Cache_types.x10 encoder;
  Pbrt.Encoder.key (2, Pbrt.Varint) encoder; 
  Pbrt.Encoder.int32_as_varint v.Cache_types.w10 encoder;
  Pbrt.Encoder.key (3, Pbrt.Varint) encoder; 
  Pbrt.Encoder.int32_as_varint v.Cache_types.h10 encoder;
  ()

let rec encode_tower_cash_entry (v:Cache_types.tower_cash_entry) encoder = 
  List.iter (fun x -> 
    Pbrt.Encoder.key (4, Pbrt.Bytes) encoder; 
    Pbrt.Encoder.nested (encode_tower_cash_block x) encoder;
  ) v.Cache_types.plan;
  Pbrt.Encoder.key (5, Pbrt.Bits32) encoder; 
  Pbrt.Encoder.float_as_bits32 v.Cache_types.height encoder;
  Pbrt.Encoder.key (6, Pbrt.Bits32) encoder; 
  Pbrt.Encoder.float_as_bits32 v.Cache_types.stability encoder;
  Pbrt.Encoder.key (7, Pbrt.Bits32) encoder; 
  Pbrt.Encoder.float_as_bits32 v.Cache_types.area encoder;
  Pbrt.Encoder.key (8, Pbrt.Bits32) encoder; 
  Pbrt.Encoder.float_as_bits32 v.Cache_types.length encoder;
  Pbrt.Encoder.key (9, Pbrt.Bits32) encoder; 
  Pbrt.Encoder.float_as_bits32 v.Cache_types.overpass encoder;
  Pbrt.Encoder.key (10, Pbrt.Bits32) encoder; 
  Pbrt.Encoder.float_as_bits32 v.Cache_types.staircase encoder;
  ()

let rec encode_tower_cash (v:Cache_types.tower_cash) encoder = 
  List.iter (fun x -> 
    Pbrt.Encoder.key (1, Pbrt.Bytes) encoder; 
    Pbrt.Encoder.nested (encode_tower_cash_entry x) encoder;
  ) v.Cache_types.entries;
  ()
