type id =
  | C
  | P of id
and point =
  | Intr of twoDType * twoDType
  | On   of twoDType (* This is a random point *on* object X *)
  | Id1  of id
  | Any
and twoDType =
  | Circ of point * point
  | Line of point * point
  | Id2  of id
and element =
  | OneD of point
  | TwoD of twoDType
and construction = End | Rest of element * construction (* (e,c) list *)

let rec reconcat l = match l with
  | [] -> End
  | x::r -> Rest(x,(reconcat r))

let eqTriangle =
  reconcat
    [ OneD(Any)                                            ;
      OneD(Any)                                            ;
      TwoD(Circ(Id1(P(P(C))),Id1(P(C))))                   ;
      TwoD(Circ(Id1(P(P(C))),Id1(P(P(P(C))))))             ;
      OneD(Intr(Id2(P(P(C))),Id2(P(C))))                   ;
      TwoD(Line(Id1(P(P(P(P(P(C)))))),Id1(P(P(P(P(C))))))) ;
      TwoD(Line(Id1(P(P(P(P(P(P(C))))))),Id1(P(P(C)))))    ;
      TwoD(Line(Id1(P(P(P(P(P(P(C))))))),Id1(P(P(P(C)))))) ]

let _ = (* Do nothing. I'm interested in the typing so far. *)
  print_endline "Success, I was typed."
