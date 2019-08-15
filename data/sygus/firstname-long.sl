(set-logic SLIA)
 
(synth-fun f ((name String)) String
    ((Start String (ntString))
     (ntString String (name " "
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
 
(constraint (= (f "Nancy FreeHafer") "Nancy"))
(constraint (= (f "Andrew Cencici") "Andrew"))
(constraint (= (f "Jan Kotas") "Jan"))
(constraint (= (f "Mariya Sergienko") "Mariya"))
(constraint (= (f "Launa Withers") "Launa"))
(constraint (= (f "Lakenya Edison") "Lakenya"))
(constraint (= (f "Brendan Hage") "Brendan"))
(constraint (= (f "Bradford Lango") "Bradford"))
(constraint (= (f "Rudolf Akiyama") "Rudolf"))
(constraint (= (f "Lara Constable") "Lara"))
(constraint (= (f "Madelaine Ghoston") "Madelaine"))
(constraint (= (f "Salley Hornak") "Salley"))
(constraint (= (f "Micha Junkin") "Micha"))
(constraint (= (f "Teddy Bobo") "Teddy"))
(constraint (= (f "Coralee Scalia") "Coralee"))
(constraint (= (f "Jeff Quashie") "Jeff"))
(constraint (= (f "Vena Babiarz") "Vena"))
(constraint (= (f "Karrie Lain") "Karrie"))
(constraint (= (f "Tobias Dermody") "Tobias"))
(constraint (= (f "Celsa Hopkins") "Celsa"))
(constraint (= (f "Kimberley Halpern") "Kimberley"))
(constraint (= (f "Phillip Rowden") "Phillip"))
(constraint (= (f "Elias Neil") "Elias"))
(constraint (= (f "Lashanda Cortes") "Lashanda"))
(constraint (= (f "Mackenzie Spell") "Mackenzie"))
(constraint (= (f "Kathlyn Eccleston") "Kathlyn"))
(constraint (= (f "Georgina Brescia") "Georgina"))
(constraint (= (f "Beata Miah") "Beata"))
(constraint (= (f "Desiree Seamons") "Desiree"))
(constraint (= (f "Jeanice Soderstrom") "Jeanice"))
(constraint (= (f "Mariel Jurgens") "Mariel"))
(constraint (= (f "Alida Bogle") "Alida"))
(constraint (= (f "Jacqualine Olague") "Jacqualine"))
(constraint (= (f "Joaquin Clasen") "Joaquin"))
(constraint (= (f "Samuel Richert") "Samuel"))
(constraint (= (f "Malissa Marcus") "Malissa"))
(constraint (= (f "Alaina Partida") "Alaina"))
(constraint (= (f "Trinidad Mulloy") "Trinidad"))
(constraint (= (f "Carlene Garrard") "Carlene"))
(constraint (= (f "Melodi Chism") "Melodi"))
(constraint (= (f "Bess Chilcott") "Bess"))
(constraint (= (f "Chong Aylward") "Chong"))
(constraint (= (f "Jani Ramthun") "Jani"))
(constraint (= (f "Jacquiline Heintz") "Jacquiline"))
(constraint (= (f "Hayley Marquess") "Hayley"))
(constraint (= (f "Andria Spagnoli") "Andria"))
(constraint (= (f "Irwin Covelli") "Irwin"))
(constraint (= (f "Gertude Montiel") "Gertude"))
(constraint (= (f "Stefany Reily") "Stefany"))
(constraint (= (f "Rae Mcgaughey") "Rae"))
(constraint (= (f "Cruz Latimore") "Cruz"))
(constraint (= (f "Maryann Casler") "Maryann"))
(constraint (= (f "Annalisa Gregori") "Annalisa"))
(constraint (= (f "Jenee Pannell") "Jenee"))
 
(check-synth)
