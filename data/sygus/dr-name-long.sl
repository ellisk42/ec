(set-logic SLIA)
 
(synth-fun f ((name String)) String
    ((Start String (ntString))
     (ntString String (name " " "." "Dr." "D" "r"
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
 
(constraint (= (f "Launa Withers") "Dr. Launa"))
(constraint (= (f "Lakenya Edison") "Dr. Lakenya"))
(constraint (= (f "Brendan Hage") "Dr. Brendan"))
(constraint (= (f "Bradford Lango") "Dr. Bradford"))
(constraint (= (f "Rudolf Akiyama") "Dr. Rudolf"))
(constraint (= (f "Lara Constable") "Dr. Lara"))
(constraint (= (f "Madelaine Ghoston") "Dr. Madelaine"))
(constraint (= (f "Salley Hornak") "Dr. Salley"))
(constraint (= (f "Micha Junkin") "Dr. Micha"))
(constraint (= (f "Teddy Bobo") "Dr. Teddy"))
(constraint (= (f "Coralee Scalia") "Dr. Coralee"))
(constraint (= (f "Jeff Quashie") "Dr. Jeff"))
(constraint (= (f "Vena Babiarz") "Dr. Vena"))
(constraint (= (f "Karrie Lain") "Dr. Karrie"))
(constraint (= (f "Tobias Dermody") "Dr. Tobias"))
(constraint (= (f "Celsa Hopkins") "Dr. Celsa"))
(constraint (= (f "Kimberley Halpern") "Dr. Kimberley"))
(constraint (= (f "Phillip Rowden") "Dr. Phillip"))
(constraint (= (f "Elias Neil") "Dr. Elias"))
(constraint (= (f "Lashanda Cortes") "Dr. Lashanda"))
(constraint (= (f "Mackenzie Spell") "Dr. Mackenzie"))
(constraint (= (f "Kathlyn Eccleston") "Dr. Kathlyn"))
(constraint (= (f "Georgina Brescia") "Dr. Georgina"))
(constraint (= (f "Beata Miah") "Dr. Beata"))
(constraint (= (f "Desiree Seamons") "Dr. Desiree"))
(constraint (= (f "Jeanice Soderstrom") "Dr. Jeanice"))
(constraint (= (f "Mariel Jurgens") "Dr. Mariel"))
(constraint (= (f "Alida Bogle") "Dr. Alida"))
(constraint (= (f "Jacqualine Olague") "Dr. Jacqualine"))
(constraint (= (f "Joaquin Clasen") "Dr. Joaquin"))
(constraint (= (f "Samuel Richert") "Dr. Samuel"))
(constraint (= (f "Malissa Marcus") "Dr. Malissa"))
(constraint (= (f "Alaina Partida") "Dr. Alaina"))
(constraint (= (f "Trinidad Mulloy") "Dr. Trinidad"))
(constraint (= (f "Carlene Garrard") "Dr. Carlene"))
(constraint (= (f "Melodi Chism") "Dr. Melodi"))
(constraint (= (f "Bess Chilcott") "Dr. Bess"))
(constraint (= (f "Chong Aylward") "Dr. Chong"))
(constraint (= (f "Jani Ramthun") "Dr. Jani"))
(constraint (= (f "Jacquiline Heintz") "Dr. Jacquiline"))
(constraint (= (f "Hayley Marquess") "Dr. Hayley"))
(constraint (= (f "Andria Spagnoli") "Dr. Andria"))
(constraint (= (f "Irwin Covelli") "Dr. Irwin"))
(constraint (= (f "Gertude Montiel") "Dr. Gertude"))
(constraint (= (f "Stefany Reily") "Dr. Stefany"))
(constraint (= (f "Rae Mcgaughey") "Dr. Rae"))
(constraint (= (f "Cruz Latimore") "Dr. Cruz"))
(constraint (= (f "Maryann Casler") "Dr. Maryann"))
(constraint (= (f "Annalisa Gregori") "Dr. Annalisa"))
(constraint (= (f "Jenee Pannell") "Dr. Jenee"))

(check-synth)
