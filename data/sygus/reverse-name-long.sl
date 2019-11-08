(set-logic SLIA)
 
(synth-fun f ((firstname String) (lastname String)) String
    ((Start String (ntString))
     (ntString String (firstname lastname " "
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

(constraint (= (f "Launa" "Withers") "Withers Launa"))
(constraint (= (f "Lakenya" "Edison") "Edison Lakenya"))
(constraint (= (f "Brendan" "Hage") "Hage Brendan"))
(constraint (= (f "Bradford" "Lango") "Lango Bradford"))
(constraint (= (f "Rudolf" "Akiyama") "Akiyama Rudolf"))
(constraint (= (f "Lara" "Constable") "Constable Lara"))
(constraint (= (f "Madelaine" "Ghoston") "Ghoston Madelaine"))
(constraint (= (f "Salley" "Hornak") "Hornak Salley"))
(constraint (= (f "Micha" "Junkin") "Junkin Micha"))
(constraint (= (f "Teddy" "Bobo") "Bobo Teddy"))
(constraint (= (f "Coralee" "Scalia") "Scalia Coralee"))
(constraint (= (f "Jeff" "Quashie") "Quashie Jeff"))
(constraint (= (f "Vena" "Babiarz") "Babiarz Vena"))
(constraint (= (f "Karrie" "Lain") "Lain Karrie"))
(constraint (= (f "Tobias" "Dermody") "Dermody Tobias"))
(constraint (= (f "Celsa" "Hopkins") "Hopkins Celsa"))
(constraint (= (f "Kimberley" "Halpern") "Halpern Kimberley"))
(constraint (= (f "Phillip" "Rowden") "Rowden Phillip"))
(constraint (= (f "Elias" "Neil") "Neil Elias"))
(constraint (= (f "Lashanda" "Cortes") "Cortes Lashanda"))
(constraint (= (f "Mackenzie" "Spell") "Spell Mackenzie"))
(constraint (= (f "Kathlyn" "Eccleston") "Eccleston Kathlyn"))
(constraint (= (f "Georgina" "Brescia") "Brescia Georgina"))
(constraint (= (f "Beata" "Miah") "Miah Beata"))
(constraint (= (f "Desiree" "Seamons") "Seamons Desiree"))
(constraint (= (f "Jeanice" "Soderstrom") "Soderstrom Jeanice"))
(constraint (= (f "Mariel" "Jurgens") "Jurgens Mariel"))
(constraint (= (f "Alida" "Bogle") "Bogle Alida"))
(constraint (= (f "Jacqualine" "Olague") "Olague Jacqualine"))
(constraint (= (f "Joaquin" "Clasen") "Clasen Joaquin"))
(constraint (= (f "Samuel" "Richert") "Richert Samuel"))
(constraint (= (f "Malissa" "Marcus") "Marcus Malissa"))
(constraint (= (f "Alaina" "Partida") "Partida Alaina"))
(constraint (= (f "Trinidad" "Mulloy") "Mulloy Trinidad"))
(constraint (= (f "Carlene" "Garrard") "Garrard Carlene"))
(constraint (= (f "Melodi" "Chism") "Chism Melodi"))
(constraint (= (f "Bess" "Chilcott") "Chilcott Bess"))
(constraint (= (f "Chong" "Aylward") "Aylward Chong"))
(constraint (= (f "Jani" "Ramthun") "Ramthun Jani"))
(constraint (= (f "Jacquiline" "Heintz") "Heintz Jacquiline"))
(constraint (= (f "Hayley" "Marquess") "Marquess Hayley"))
(constraint (= (f "Andria" "Spagnoli") "Spagnoli Andria"))
(constraint (= (f "Irwin" "Covelli") "Covelli Irwin"))
(constraint (= (f "Gertude" "Montiel") "Montiel Gertude"))
(constraint (= (f "Stefany" "Reily") "Reily Stefany"))
(constraint (= (f "Rae" "Mcgaughey") "Mcgaughey Rae"))
(constraint (= (f "Cruz" "Latimore") "Latimore Cruz"))
(constraint (= (f "Maryann" "Casler") "Casler Maryann"))
(constraint (= (f "Annalisa" "Gregori") "Gregori Annalisa"))
(constraint (= (f "Jenee" "Pannell") "Pannell Jenee"))
 
(check-synth)
