(set-logic SLIA)
 
(synth-fun f ((name String)) String
    ((Start String (ntString))
     (ntString String (name " " 
                       (str.++ ntString ntString)
                       (str.replace ntString ntString ntString)
                       (str.at ntString ntInt)
                       (int.to.str ntInt)
                       (str.substr ntString ntInt ntInt)))
      (ntInt Int (0 1
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
 
(constraint (= (f "Nancy FreeHafer") "FreeHafer"))
(constraint (= (f "Andrew Cencici") "Cencici"))
(constraint (= (f "Jan Kotas") "Kotas"))
(constraint (= (f "Mariya Sergienko") "Sergienko"))
(constraint (= (f "Launa Withers") "Withers"))
(constraint (= (f "Lakenya Edison") "Edison"))
(constraint (= (f "Brendan Hage") "Hage"))
(constraint (= (f "Bradford Lango") "Lango"))
(constraint (= (f "Rudolf Akiyama") "Akiyama"))
(constraint (= (f "Lara Constable") "Constable"))
(constraint (= (f "Madelaine Ghoston") "Ghoston"))
(constraint (= (f "Salley Hornak") "Hornak"))
(constraint (= (f "Micha Junkin") "Junkin"))
(constraint (= (f "Teddy Bobo") "Bobo"))
(constraint (= (f "Coralee Scalia") "Scalia"))
(constraint (= (f "Jeff Quashie") "Quashie"))
(constraint (= (f "Vena Babiarz") "Babiarz"))
(constraint (= (f "Karrie Lain") "Lain"))
(constraint (= (f "Tobias Dermody") "Dermody"))
(constraint (= (f "Celsa Hopkins") "Hopkins"))
(constraint (= (f "Kimberley Halpern") "Halpern"))
(constraint (= (f "Phillip Rowden") "Rowden"))
(constraint (= (f "Elias Neil") "Neil"))
(constraint (= (f "Lashanda Cortes") "Cortes"))
(constraint (= (f "Mackenzie Spell") "Spell"))
(constraint (= (f "Kathlyn Eccleston") "Eccleston"))
(constraint (= (f "Georgina Brescia") "Brescia"))
(constraint (= (f "Beata Miah") "Miah"))
(constraint (= (f "Desiree Seamons") "Seamons"))
(constraint (= (f "Jeanice Soderstrom") "Soderstrom"))
(constraint (= (f "Mariel Jurgens") "Jurgens"))
(constraint (= (f "Alida Bogle") "Bogle"))
(constraint (= (f "Jacqualine Olague") "Olague"))
(constraint (= (f "Joaquin Clasen") "Clasen"))
(constraint (= (f "Samuel Richert") "Richert"))
(constraint (= (f "Malissa Marcus") "Marcus"))
(constraint (= (f "Alaina Partida") "Partida"))
(constraint (= (f "Trinidad Mulloy") "Mulloy"))
(constraint (= (f "Carlene Garrard") "Garrard"))
(constraint (= (f "Melodi Chism") "Chism"))
(constraint (= (f "Bess Chilcott") "Chilcott"))
(constraint (= (f "Chong Aylward") "Aylward"))
(constraint (= (f "Jani Ramthun") "Ramthun"))
(constraint (= (f "Jacquiline Heintz") "Heintz"))
(constraint (= (f "Hayley Marquess") "Marquess"))
(constraint (= (f "Andria Spagnoli") "Spagnoli"))
(constraint (= (f "Irwin Covelli") "Covelli"))
(constraint (= (f "Gertude Montiel") "Montiel"))
(constraint (= (f "Stefany Reily") "Reily"))
(constraint (= (f "Rae Mcgaughey") "Mcgaughey"))
(constraint (= (f "Cruz Latimore") "Latimore"))
(constraint (= (f "Maryann Casler") "Casler"))
(constraint (= (f "Annalisa Gregori") "Gregori"))
(constraint (= (f "Jenee Pannell") "Pannell"))
 
(check-synth)
