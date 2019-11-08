(set-logic SLIA)
 
(synth-fun f ((firstname String) (lastname String)) String
    ((Start String (ntString))
     (ntString String (firstname lastname " " "."
                       (str.++ ntString ntString)
                       (str.replace ntString ntString ntString)
                       (str.at ntString ntInt)
                       (int.to.str ntInt)
                       (str.substr ntString ntInt ntInt)))
      (ntInt Int (0 1 2
                  (+ ntInt ntInt)
                  (- ntInt ntInt)
                  (str.len ntString)
                  (str.to.int ntString)
                  (str.indexof ntString ntString ntInt)))
      (ntBool Bool (true false
                    (str.prefixof ntString ntString)
                    (str.suffixof ntString ntString)
                    (str.contains ntString ntString)))))


(declare-var firstname String)
(declare-var lastname String)

 
(constraint (= (f "Nancy" "FreeHafer") "Nancy F."))
(constraint (= (f "Andrew" "Cencici") "Andrew C."))
(constraint (= (f "Jan" "Kotas") "Jan K."))
(constraint (= (f "Mariya" "Sergienko") "Mariya S."))
(constraint (= (f "Launa" "Withers") "Launa W."))
(constraint (= (f "Lakenya" "Edison") "Lakenya E."))
(constraint (= (f "Brendan" "Hage") "Brendan H."))
(constraint (= (f "Bradford" "Lango") "Bradford L."))
(constraint (= (f "Rudolf" "Akiyama") "Rudolf A."))
(constraint (= (f "Lara" "Constable") "Lara C."))
(constraint (= (f "Madelaine" "Ghoston") "Madelaine G."))
(constraint (= (f "Salley" "Hornak") "Salley H."))
(constraint (= (f "Micha" "Junkin") "Micha J."))
(constraint (= (f "Teddy" "Bobo") "Teddy B."))
(constraint (= (f "Coralee" "Scalia") "Coralee S."))
(constraint (= (f "Jeff" "Quashie") "Jeff Q."))
(constraint (= (f "Vena" "Babiarz") "Vena B."))
(constraint (= (f "Karrie" "Lain") "Karrie L."))
(constraint (= (f "Tobias" "Dermody") "Tobias D."))
(constraint (= (f "Celsa" "Hopkins") "Celsa H."))
(constraint (= (f "Kimberley" "Halpern") "Kimberley H."))
(constraint (= (f "Phillip" "Rowden") "Phillip R."))
(constraint (= (f "Elias" "Neil") "Elias N."))
(constraint (= (f "Lashanda" "Cortes") "Lashanda C."))
(constraint (= (f "Mackenzie" "Spell") "Mackenzie S."))
(constraint (= (f "Kathlyn" "Eccleston") "Kathlyn E."))
(constraint (= (f "Georgina" "Brescia") "Georgina B."))
(constraint (= (f "Beata" "Miah") "Beata M."))
(constraint (= (f "Desiree" "Seamons") "Desiree S."))
(constraint (= (f "Jeanice" "Soderstrom") "Jeanice S."))
(constraint (= (f "Mariel" "Jurgens") "Mariel J."))
(constraint (= (f "Alida" "Bogle") "Alida B."))
(constraint (= (f "Jacqualine" "Olague") "Jacqualine O."))
(constraint (= (f "Joaquin" "Clasen") "Joaquin C."))
(constraint (= (f "Samuel" "Richert") "Samuel R."))
(constraint (= (f "Malissa" "Marcus") "Malissa M."))
(constraint (= (f "Alaina" "Partida") "Alaina P."))
(constraint (= (f "Trinidad" "Mulloy") "Trinidad M."))
(constraint (= (f "Carlene" "Garrard") "Carlene G."))
(constraint (= (f "Melodi" "Chism") "Melodi C."))
(constraint (= (f "Bess" "Chilcott") "Bess C."))
(constraint (= (f "Chong" "Aylward") "Chong A."))
(constraint (= (f "Jani" "Ramthun") "Jani R."))
(constraint (= (f "Jacquiline" "Heintz") "Jacquiline H."))
(constraint (= (f "Hayley" "Marquess") "Hayley M."))
(constraint (= (f "Andria" "Spagnoli") "Andria S."))
(constraint (= (f "Irwin" "Covelli") "Irwin C."))
(constraint (= (f "Gertude" "Montiel") "Gertude M."))
(constraint (= (f "Stefany" "Reily") "Stefany R."))
(constraint (= (f "Rae" "Mcgaughey") "Rae M."))
(constraint (= (f "Cruz" "Latimore") "Cruz L."))
(constraint (= (f "Maryann" "Casler") "Maryann C."))
(constraint (= (f "Annalisa" "Gregori") "Annalisa G."))
(constraint (= (f "Jenee" "Pannell") "Jenee P."))
 
(check-synth)
