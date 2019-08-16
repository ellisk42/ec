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

(constraint (= (f "Launa" "Withers") "L. Withers"))
(constraint (= (f "Launa" "Withers") "L. Withers"))
(constraint (= (f "Launa" "Withers") "L. Withers"))
(constraint (= (f "Lakenya" "Edison") "L. Edison"))
(constraint (= (f "Lakenya" "Edison") "L. Edison"))
(constraint (= (f "Lakenya" "Edison") "L. Edison"))
(constraint (= (f "Brendan" "Hage") "B. Hage"))
(constraint (= (f "Brendan" "Hage") "B. Hage"))
(constraint (= (f "Brendan" "Hage") "B. Hage"))
(constraint (= (f "Bradford" "Lango") "B. Lango"))
(constraint (= (f "Bradford" "Lango") "B. Lango"))
(constraint (= (f "Bradford" "Lango") "B. Lango"))
(constraint (= (f "Rudolf" "Akiyama") "R. Akiyama"))
(constraint (= (f "Rudolf" "Akiyama") "R. Akiyama"))
(constraint (= (f "Rudolf" "Akiyama") "R. Akiyama"))
(constraint (= (f "Lara" "Constable") "L. Constable"))
(constraint (= (f "Lara" "Constable") "L. Constable"))
(constraint (= (f "Lara" "Constable") "L. Constable"))
(constraint (= (f "Madelaine" "Ghoston") "M. Ghoston"))
(constraint (= (f "Madelaine" "Ghoston") "M. Ghoston"))
(constraint (= (f "Madelaine" "Ghoston") "M. Ghoston"))
(constraint (= (f "Salley" "Hornak") "S. Hornak"))
(constraint (= (f "Salley" "Hornak") "S. Hornak"))
(constraint (= (f "Salley" "Hornak") "S. Hornak"))
(constraint (= (f "Micha" "Junkin") "M. Junkin"))
(constraint (= (f "Micha" "Junkin") "M. Junkin"))
(constraint (= (f "Micha" "Junkin") "M. Junkin"))
(constraint (= (f "Teddy" "Bobo") "T. Bobo"))
(constraint (= (f "Teddy" "Bobo") "T. Bobo"))
(constraint (= (f "Teddy" "Bobo") "T. Bobo"))
(constraint (= (f "Coralee" "Scalia") "C. Scalia"))
(constraint (= (f "Coralee" "Scalia") "C. Scalia"))
(constraint (= (f "Coralee" "Scalia") "C. Scalia"))
(constraint (= (f "Jeff" "Quashie") "J. Quashie"))
(constraint (= (f "Jeff" "Quashie") "J. Quashie"))
(constraint (= (f "Jeff" "Quashie") "J. Quashie"))
(constraint (= (f "Vena" "Babiarz") "V. Babiarz"))
(constraint (= (f "Vena" "Babiarz") "V. Babiarz"))
(constraint (= (f "Vena" "Babiarz") "V. Babiarz"))
(constraint (= (f "Karrie" "Lain") "K. Lain"))
(constraint (= (f "Karrie" "Lain") "K. Lain"))
(constraint (= (f "Karrie" "Lain") "K. Lain"))
(constraint (= (f "Tobias" "Dermody") "T. Dermody"))
(constraint (= (f "Tobias" "Dermody") "T. Dermody"))
(constraint (= (f "Tobias" "Dermody") "T. Dermody"))
(constraint (= (f "Celsa" "Hopkins") "C. Hopkins"))
(constraint (= (f "Celsa" "Hopkins") "C. Hopkins"))
(constraint (= (f "Celsa" "Hopkins") "C. Hopkins"))
(constraint (= (f "Kimberley" "Halpern") "K. Halpern"))
(constraint (= (f "Kimberley" "Halpern") "K. Halpern"))
(constraint (= (f "Kimberley" "Halpern") "K. Halpern"))
(constraint (= (f "Phillip" "Rowden") "P. Rowden"))
(constraint (= (f "Phillip" "Rowden") "P. Rowden"))
(constraint (= (f "Phillip" "Rowden") "P. Rowden"))
(constraint (= (f "Elias" "Neil") "E. Neil"))
(constraint (= (f "Elias" "Neil") "E. Neil"))
(constraint (= (f "Elias" "Neil") "E. Neil"))
(constraint (= (f "Lashanda" "Cortes") "L. Cortes"))
(constraint (= (f "Lashanda" "Cortes") "L. Cortes"))
(constraint (= (f "Lashanda" "Cortes") "L. Cortes"))
(constraint (= (f "Mackenzie" "Spell") "M. Spell"))
(constraint (= (f "Mackenzie" "Spell") "M. Spell"))
(constraint (= (f "Mackenzie" "Spell") "M. Spell"))
(constraint (= (f "Kathlyn" "Eccleston") "K. Eccleston"))
(constraint (= (f "Kathlyn" "Eccleston") "K. Eccleston"))
(constraint (= (f "Kathlyn" "Eccleston") "K. Eccleston"))
(constraint (= (f "Georgina" "Brescia") "G. Brescia"))
(constraint (= (f "Georgina" "Brescia") "G. Brescia"))
(constraint (= (f "Georgina" "Brescia") "G. Brescia"))
(constraint (= (f "Beata" "Miah") "B. Miah"))
(constraint (= (f "Beata" "Miah") "B. Miah"))
(constraint (= (f "Beata" "Miah") "B. Miah"))
(constraint (= (f "Desiree" "Seamons") "D. Seamons"))
(constraint (= (f "Desiree" "Seamons") "D. Seamons"))
(constraint (= (f "Desiree" "Seamons") "D. Seamons"))
(constraint (= (f "Jeanice" "Soderstrom") "J. Soderstrom"))
(constraint (= (f "Jeanice" "Soderstrom") "J. Soderstrom"))
(constraint (= (f "Jeanice" "Soderstrom") "J. Soderstrom"))
(constraint (= (f "Mariel" "Jurgens") "M. Jurgens"))
(constraint (= (f "Mariel" "Jurgens") "M. Jurgens"))
(constraint (= (f "Mariel" "Jurgens") "M. Jurgens"))
(constraint (= (f "Alida" "Bogle") "A. Bogle"))
(constraint (= (f "Alida" "Bogle") "A. Bogle"))
(constraint (= (f "Alida" "Bogle") "A. Bogle"))
(constraint (= (f "Jacqualine" "Olague") "J. Olague"))
(constraint (= (f "Jacqualine" "Olague") "J. Olague"))
(constraint (= (f "Jacqualine" "Olague") "J. Olague"))
(constraint (= (f "Joaquin" "Clasen") "J. Clasen"))
(constraint (= (f "Joaquin" "Clasen") "J. Clasen"))
(constraint (= (f "Joaquin" "Clasen") "J. Clasen"))
(constraint (= (f "Samuel" "Richert") "S. Richert"))
(constraint (= (f "Samuel" "Richert") "S. Richert"))
(constraint (= (f "Samuel" "Richert") "S. Richert"))
(constraint (= (f "Malissa" "Marcus") "M. Marcus"))
(constraint (= (f "Malissa" "Marcus") "M. Marcus"))
(constraint (= (f "Malissa" "Marcus") "M. Marcus"))
(constraint (= (f "Alaina" "Partida") "A. Partida"))
(constraint (= (f "Alaina" "Partida") "A. Partida"))
(constraint (= (f "Alaina" "Partida") "A. Partida"))
(constraint (= (f "Trinidad" "Mulloy") "T. Mulloy"))
(constraint (= (f "Trinidad" "Mulloy") "T. Mulloy"))
(constraint (= (f "Trinidad" "Mulloy") "T. Mulloy"))
(constraint (= (f "Carlene" "Garrard") "C. Garrard"))
(constraint (= (f "Carlene" "Garrard") "C. Garrard"))
(constraint (= (f "Carlene" "Garrard") "C. Garrard"))
(constraint (= (f "Melodi" "Chism") "M. Chism"))
(constraint (= (f "Melodi" "Chism") "M. Chism"))
(constraint (= (f "Melodi" "Chism") "M. Chism"))
(constraint (= (f "Bess" "Chilcott") "B. Chilcott"))
(constraint (= (f "Bess" "Chilcott") "B. Chilcott"))
(constraint (= (f "Bess" "Chilcott") "B. Chilcott"))
(constraint (= (f "Chong" "Aylward") "C. Aylward"))
(constraint (= (f "Chong" "Aylward") "C. Aylward"))
(constraint (= (f "Chong" "Aylward") "C. Aylward"))
(constraint (= (f "Jani" "Ramthun") "J. Ramthun"))
(constraint (= (f "Jani" "Ramthun") "J. Ramthun"))
(constraint (= (f "Jani" "Ramthun") "J. Ramthun"))
(constraint (= (f "Jacquiline" "Heintz") "J. Heintz"))
(constraint (= (f "Jacquiline" "Heintz") "J. Heintz"))
(constraint (= (f "Jacquiline" "Heintz") "J. Heintz"))
(constraint (= (f "Hayley" "Marquess") "H. Marquess"))
(constraint (= (f "Hayley" "Marquess") "H. Marquess"))
(constraint (= (f "Hayley" "Marquess") "H. Marquess"))
(constraint (= (f "Andria" "Spagnoli") "A. Spagnoli"))
(constraint (= (f "Andria" "Spagnoli") "A. Spagnoli"))
(constraint (= (f "Andria" "Spagnoli") "A. Spagnoli"))
(constraint (= (f "Irwin" "Covelli") "I. Covelli"))
(constraint (= (f "Irwin" "Covelli") "I. Covelli"))
(constraint (= (f "Irwin" "Covelli") "I. Covelli"))
(constraint (= (f "Gertude" "Montiel") "G. Montiel"))
(constraint (= (f "Gertude" "Montiel") "G. Montiel"))
(constraint (= (f "Gertude" "Montiel") "G. Montiel"))
(constraint (= (f "Stefany" "Reily") "S. Reily"))
(constraint (= (f "Stefany" "Reily") "S. Reily"))
(constraint (= (f "Stefany" "Reily") "S. Reily"))
(constraint (= (f "Rae" "Mcgaughey") "R. Mcgaughey"))
(constraint (= (f "Rae" "Mcgaughey") "R. Mcgaughey"))
(constraint (= (f "Rae" "Mcgaughey") "R. Mcgaughey"))
(constraint (= (f "Cruz" "Latimore") "C. Latimore"))
(constraint (= (f "Cruz" "Latimore") "C. Latimore"))
(constraint (= (f "Cruz" "Latimore") "C. Latimore"))
(constraint (= (f "Maryann" "Casler") "M. Casler"))
(constraint (= (f "Maryann" "Casler") "M. Casler"))
(constraint (= (f "Maryann" "Casler") "M. Casler"))
(constraint (= (f "Annalisa" "Gregori") "A. Gregori"))
(constraint (= (f "Annalisa" "Gregori") "A. Gregori"))
(constraint (= (f "Annalisa" "Gregori") "A. Gregori"))
(constraint (= (f "Jenee" "Pannell") "J. Pannell"))
(constraint (= (f "Jenee" "Pannell") "J. Pannell"))
(constraint (= (f "Jenee" "Pannell") "J. Pannell"))
(constraint (= (f "Launa" "Withers") "L. Withers"))
(constraint (= (f "Lakenya" "Edison") "L. Edison"))
(constraint (= (f "Brendan" "Hage") "B. Hage"))
(constraint (= (f "Bradford" "Lango") "B. Lango"))
(constraint (= (f "Rudolf" "Akiyama") "R. Akiyama"))
(constraint (= (f "Lara" "Constable") "L. Constable"))
(constraint (= (f "Madelaine" "Ghoston") "M. Ghoston"))
(constraint (= (f "Salley" "Hornak") "S. Hornak"))
(constraint (= (f "Micha" "Junkin") "M. Junkin"))
(constraint (= (f "Teddy" "Bobo") "T. Bobo"))
(constraint (= (f "Coralee" "Scalia") "C. Scalia"))
(constraint (= (f "Jeff" "Quashie") "J. Quashie"))
(constraint (= (f "Vena" "Babiarz") "V. Babiarz"))
(constraint (= (f "Karrie" "Lain") "K. Lain"))
(constraint (= (f "Tobias" "Dermody") "T. Dermody"))
(constraint (= (f "Celsa" "Hopkins") "C. Hopkins"))
(constraint (= (f "Kimberley" "Halpern") "K. Halpern"))
(constraint (= (f "Phillip" "Rowden") "P. Rowden"))
(constraint (= (f "Elias" "Neil") "E. Neil"))
(constraint (= (f "Lashanda" "Cortes") "L. Cortes"))
(constraint (= (f "Mackenzie" "Spell") "M. Spell"))
(constraint (= (f "Kathlyn" "Eccleston") "K. Eccleston"))
(constraint (= (f "Georgina" "Brescia") "G. Brescia"))
(constraint (= (f "Beata" "Miah") "B. Miah"))
(constraint (= (f "Desiree" "Seamons") "D. Seamons"))
(constraint (= (f "Jeanice" "Soderstrom") "J. Soderstrom"))
(constraint (= (f "Mariel" "Jurgens") "M. Jurgens"))
(constraint (= (f "Alida" "Bogle") "A. Bogle"))
(constraint (= (f "Jacqualine" "Olague") "J. Olague"))
(constraint (= (f "Joaquin" "Clasen") "J. Clasen"))
(constraint (= (f "Samuel" "Richert") "S. Richert"))
(constraint (= (f "Malissa" "Marcus") "M. Marcus"))
(constraint (= (f "Alaina" "Partida") "A. Partida"))
(constraint (= (f "Trinidad" "Mulloy") "T. Mulloy"))
(constraint (= (f "Carlene" "Garrard") "C. Garrard"))
(constraint (= (f "Melodi" "Chism") "M. Chism"))
(constraint (= (f "Bess" "Chilcott") "B. Chilcott"))
(constraint (= (f "Chong" "Aylward") "C. Aylward"))
(constraint (= (f "Jani" "Ramthun") "J. Ramthun"))
(constraint (= (f "Jacquiline" "Heintz") "J. Heintz"))
(constraint (= (f "Hayley" "Marquess") "H. Marquess"))
(constraint (= (f "Andria" "Spagnoli") "A. Spagnoli"))
(constraint (= (f "Irwin" "Covelli") "I. Covelli"))
(constraint (= (f "Gertude" "Montiel") "G. Montiel"))
(constraint (= (f "Stefany" "Reily") "S. Reily"))
(constraint (= (f "Rae" "Mcgaughey") "R. Mcgaughey"))
(constraint (= (f "Cruz" "Latimore") "C. Latimore"))
(constraint (= (f "Maryann" "Casler") "M. Casler"))
(constraint (= (f "Annalisa" "Gregori") "A. Gregori"))
(constraint (= (f "Jenee" "Pannell") "J. Pannell"))

 
(check-synth)
