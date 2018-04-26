(set-logic SLIA)
 
(synth-fun f ((name String)) String
    ((Start String (ntString))
     (ntString String (name " " "."
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


(declare-var name String)
 
(constraint (= (f "Nancy FreeHafer") "N.F."))
(constraint (= (f "Andrew Cencici") "A.C."))
(constraint (= (f "Jan Kotas") "J.K."))
(constraint (= (f "Mariya Sergienko") "M.S."))
(constraint (= (f "Launa Withers") "L.W."))
(constraint (= (f "Lakenya Edison") "L.E."))
(constraint (= (f "Brendan Hage") "B.H."))
(constraint (= (f "Bradford Lango") "B.L."))
(constraint (= (f "Rudolf Akiyama") "R.A."))
(constraint (= (f "Lara Constable") "L.C."))
(constraint (= (f "Madelaine Ghoston") "M.G."))
(constraint (= (f "Salley Hornak") "S.H."))
(constraint (= (f "Micha Junkin") "M.J."))
(constraint (= (f "Teddy Bobo") "T.B."))
(constraint (= (f "Coralee Scalia") "C.S."))
(constraint (= (f "Jeff Quashie") "J.Q."))
(constraint (= (f "Vena Babiarz") "V.B."))
(constraint (= (f "Karrie Lain") "K.L."))
(constraint (= (f "Tobias Dermody") "T.D."))
(constraint (= (f "Celsa Hopkins") "C.H."))
(constraint (= (f "Kimberley Halpern") "K.H."))
(constraint (= (f "Phillip Rowden") "P.R."))
(constraint (= (f "Elias Neil") "E.N."))
(constraint (= (f "Lashanda Cortes") "L.C."))
(constraint (= (f "Mackenzie Spell") "M.S."))
(constraint (= (f "Kathlyn Eccleston") "K.E."))
(constraint (= (f "Georgina Brescia") "G.B."))
(constraint (= (f "Beata Miah") "B.M."))
(constraint (= (f "Desiree Seamons") "D.S."))
(constraint (= (f "Jeanice Soderstrom") "J.S."))
(constraint (= (f "Mariel Jurgens") "M.J."))
(constraint (= (f "Alida Bogle") "A.B."))
(constraint (= (f "Jacqualine Olague") "J.O."))
(constraint (= (f "Joaquin Clasen") "J.C."))
(constraint (= (f "Samuel Richert") "S.R."))
(constraint (= (f "Malissa Marcus") "M.M."))
(constraint (= (f "Alaina Partida") "A.P."))
(constraint (= (f "Trinidad Mulloy") "T.M."))
(constraint (= (f "Carlene Garrard") "C.G."))
(constraint (= (f "Melodi Chism") "M.C."))
(constraint (= (f "Bess Chilcott") "B.C."))
(constraint (= (f "Chong Aylward") "C.A."))
(constraint (= (f "Jani Ramthun") "J.R."))
(constraint (= (f "Jacquiline Heintz") "J.H."))
(constraint (= (f "Hayley Marquess") "H.M."))
(constraint (= (f "Andria Spagnoli") "A.S."))
(constraint (= (f "Irwin Covelli") "I.C."))
(constraint (= (f "Gertude Montiel") "G.M."))
(constraint (= (f "Stefany Reily") "S.R."))
(constraint (= (f "Rae Mcgaughey") "R.M."))
(constraint (= (f "Cruz Latimore") "C.L."))
(constraint (= (f "Maryann Casler") "M.C."))
(constraint (= (f "Annalisa Gregori") "A.G."))
(constraint (= (f "Jenee Pannell") "J.P."))

(check-synth)
