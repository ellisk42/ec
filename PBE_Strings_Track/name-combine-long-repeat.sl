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


(declare-var name String)
 
(constraint (= (f "Nancy" "FreeHafer") "Nancy FreeHafer"))
(constraint (= (f "Andrew" "Cencici") "Andrew Cencici"))
(constraint (= (f "Jan" "Kotas") "Jan Kotas"))
(constraint (= (f "Mariya" "Sergienko") "Mariya Sergienko"))
(constraint (= (f "Launa" "Withers") "Launa Withers"))
(constraint (= (f "Launa" "Withers") "Launa Withers"))
(constraint (= (f "Launa" "Withers") "Launa Withers"))
(constraint (= (f "Lakenya" "Edison") "Lakenya Edison"))
(constraint (= (f "Lakenya" "Edison") "Lakenya Edison"))
(constraint (= (f "Lakenya" "Edison") "Lakenya Edison"))
(constraint (= (f "Brendan" "Hage") "Brendan Hage"))
(constraint (= (f "Brendan" "Hage") "Brendan Hage"))
(constraint (= (f "Brendan" "Hage") "Brendan Hage"))
(constraint (= (f "Bradford" "Lango") "Bradford Lango"))
(constraint (= (f "Bradford" "Lango") "Bradford Lango"))
(constraint (= (f "Bradford" "Lango") "Bradford Lango"))
(constraint (= (f "Rudolf" "Akiyama") "Rudolf Akiyama"))
(constraint (= (f "Rudolf" "Akiyama") "Rudolf Akiyama"))
(constraint (= (f "Rudolf" "Akiyama") "Rudolf Akiyama"))
(constraint (= (f "Lara" "Constable") "Lara Constable"))
(constraint (= (f "Lara" "Constable") "Lara Constable"))
(constraint (= (f "Lara" "Constable") "Lara Constable"))
(constraint (= (f "Madelaine" "Ghoston") "Madelaine Ghoston"))
(constraint (= (f "Madelaine" "Ghoston") "Madelaine Ghoston"))
(constraint (= (f "Madelaine" "Ghoston") "Madelaine Ghoston"))
(constraint (= (f "Salley" "Hornak") "Salley Hornak"))
(constraint (= (f "Salley" "Hornak") "Salley Hornak"))
(constraint (= (f "Salley" "Hornak") "Salley Hornak"))
(constraint (= (f "Micha" "Junkin") "Micha Junkin"))
(constraint (= (f "Micha" "Junkin") "Micha Junkin"))
(constraint (= (f "Micha" "Junkin") "Micha Junkin"))
(constraint (= (f "Teddy" "Bobo") "Teddy Bobo"))
(constraint (= (f "Teddy" "Bobo") "Teddy Bobo"))
(constraint (= (f "Teddy" "Bobo") "Teddy Bobo"))
(constraint (= (f "Coralee" "Scalia") "Coralee Scalia"))
(constraint (= (f "Coralee" "Scalia") "Coralee Scalia"))
(constraint (= (f "Coralee" "Scalia") "Coralee Scalia"))
(constraint (= (f "Jeff" "Quashie") "Jeff Quashie"))
(constraint (= (f "Jeff" "Quashie") "Jeff Quashie"))
(constraint (= (f "Jeff" "Quashie") "Jeff Quashie"))
(constraint (= (f "Vena" "Babiarz") "Vena Babiarz"))
(constraint (= (f "Vena" "Babiarz") "Vena Babiarz"))
(constraint (= (f "Vena" "Babiarz") "Vena Babiarz"))
(constraint (= (f "Karrie" "Lain") "Karrie Lain"))
(constraint (= (f "Karrie" "Lain") "Karrie Lain"))
(constraint (= (f "Karrie" "Lain") "Karrie Lain"))
(constraint (= (f "Tobias" "Dermody") "Tobias Dermody"))
(constraint (= (f "Tobias" "Dermody") "Tobias Dermody"))
(constraint (= (f "Tobias" "Dermody") "Tobias Dermody"))
(constraint (= (f "Celsa" "Hopkins") "Celsa Hopkins"))
(constraint (= (f "Celsa" "Hopkins") "Celsa Hopkins"))
(constraint (= (f "Celsa" "Hopkins") "Celsa Hopkins"))
(constraint (= (f "Kimberley" "Halpern") "Kimberley Halpern"))
(constraint (= (f "Kimberley" "Halpern") "Kimberley Halpern"))
(constraint (= (f "Kimberley" "Halpern") "Kimberley Halpern"))
(constraint (= (f "Phillip" "Rowden") "Phillip Rowden"))
(constraint (= (f "Phillip" "Rowden") "Phillip Rowden"))
(constraint (= (f "Phillip" "Rowden") "Phillip Rowden"))
(constraint (= (f "Elias" "Neil") "Elias Neil"))
(constraint (= (f "Elias" "Neil") "Elias Neil"))
(constraint (= (f "Elias" "Neil") "Elias Neil"))
(constraint (= (f "Lashanda" "Cortes") "Lashanda Cortes"))
(constraint (= (f "Lashanda" "Cortes") "Lashanda Cortes"))
(constraint (= (f "Lashanda" "Cortes") "Lashanda Cortes"))
(constraint (= (f "Mackenzie" "Spell") "Mackenzie Spell"))
(constraint (= (f "Mackenzie" "Spell") "Mackenzie Spell"))
(constraint (= (f "Mackenzie" "Spell") "Mackenzie Spell"))
(constraint (= (f "Kathlyn" "Eccleston") "Kathlyn Eccleston"))
(constraint (= (f "Kathlyn" "Eccleston") "Kathlyn Eccleston"))
(constraint (= (f "Kathlyn" "Eccleston") "Kathlyn Eccleston"))
(constraint (= (f "Georgina" "Brescia") "Georgina Brescia"))
(constraint (= (f "Georgina" "Brescia") "Georgina Brescia"))
(constraint (= (f "Georgina" "Brescia") "Georgina Brescia"))
(constraint (= (f "Beata" "Miah") "Beata Miah"))
(constraint (= (f "Beata" "Miah") "Beata Miah"))
(constraint (= (f "Beata" "Miah") "Beata Miah"))
(constraint (= (f "Desiree" "Seamons") "Desiree Seamons"))
(constraint (= (f "Desiree" "Seamons") "Desiree Seamons"))
(constraint (= (f "Desiree" "Seamons") "Desiree Seamons"))
(constraint (= (f "Jeanice" "Soderstrom") "Jeanice Soderstrom"))
(constraint (= (f "Jeanice" "Soderstrom") "Jeanice Soderstrom"))
(constraint (= (f "Jeanice" "Soderstrom") "Jeanice Soderstrom"))
(constraint (= (f "Mariel" "Jurgens") "Mariel Jurgens"))
(constraint (= (f "Mariel" "Jurgens") "Mariel Jurgens"))
(constraint (= (f "Mariel" "Jurgens") "Mariel Jurgens"))
(constraint (= (f "Alida" "Bogle") "Alida Bogle"))
(constraint (= (f "Alida" "Bogle") "Alida Bogle"))
(constraint (= (f "Alida" "Bogle") "Alida Bogle"))
(constraint (= (f "Jacqualine" "Olague") "Jacqualine Olague"))
(constraint (= (f "Jacqualine" "Olague") "Jacqualine Olague"))
(constraint (= (f "Jacqualine" "Olague") "Jacqualine Olague"))
(constraint (= (f "Joaquin" "Clasen") "Joaquin Clasen"))
(constraint (= (f "Joaquin" "Clasen") "Joaquin Clasen"))
(constraint (= (f "Joaquin" "Clasen") "Joaquin Clasen"))
(constraint (= (f "Samuel" "Richert") "Samuel Richert"))
(constraint (= (f "Samuel" "Richert") "Samuel Richert"))
(constraint (= (f "Samuel" "Richert") "Samuel Richert"))
(constraint (= (f "Malissa" "Marcus") "Malissa Marcus"))
(constraint (= (f "Malissa" "Marcus") "Malissa Marcus"))
(constraint (= (f "Malissa" "Marcus") "Malissa Marcus"))
(constraint (= (f "Alaina" "Partida") "Alaina Partida"))
(constraint (= (f "Alaina" "Partida") "Alaina Partida"))
(constraint (= (f "Alaina" "Partida") "Alaina Partida"))
(constraint (= (f "Trinidad" "Mulloy") "Trinidad Mulloy"))
(constraint (= (f "Trinidad" "Mulloy") "Trinidad Mulloy"))
(constraint (= (f "Trinidad" "Mulloy") "Trinidad Mulloy"))
(constraint (= (f "Carlene" "Garrard") "Carlene Garrard"))
(constraint (= (f "Carlene" "Garrard") "Carlene Garrard"))
(constraint (= (f "Carlene" "Garrard") "Carlene Garrard"))
(constraint (= (f "Melodi" "Chism") "Melodi Chism"))
(constraint (= (f "Melodi" "Chism") "Melodi Chism"))
(constraint (= (f "Melodi" "Chism") "Melodi Chism"))
(constraint (= (f "Bess" "Chilcott") "Bess Chilcott"))
(constraint (= (f "Bess" "Chilcott") "Bess Chilcott"))
(constraint (= (f "Bess" "Chilcott") "Bess Chilcott"))
(constraint (= (f "Chong" "Aylward") "Chong Aylward"))
(constraint (= (f "Chong" "Aylward") "Chong Aylward"))
(constraint (= (f "Chong" "Aylward") "Chong Aylward"))
(constraint (= (f "Jani" "Ramthun") "Jani Ramthun"))
(constraint (= (f "Jani" "Ramthun") "Jani Ramthun"))
(constraint (= (f "Jani" "Ramthun") "Jani Ramthun"))
(constraint (= (f "Jacquiline" "Heintz") "Jacquiline Heintz"))
(constraint (= (f "Jacquiline" "Heintz") "Jacquiline Heintz"))
(constraint (= (f "Jacquiline" "Heintz") "Jacquiline Heintz"))
(constraint (= (f "Hayley" "Marquess") "Hayley Marquess"))
(constraint (= (f "Hayley" "Marquess") "Hayley Marquess"))
(constraint (= (f "Hayley" "Marquess") "Hayley Marquess"))
(constraint (= (f "Andria" "Spagnoli") "Andria Spagnoli"))
(constraint (= (f "Andria" "Spagnoli") "Andria Spagnoli"))
(constraint (= (f "Andria" "Spagnoli") "Andria Spagnoli"))
(constraint (= (f "Irwin" "Covelli") "Irwin Covelli"))
(constraint (= (f "Irwin" "Covelli") "Irwin Covelli"))
(constraint (= (f "Irwin" "Covelli") "Irwin Covelli"))
(constraint (= (f "Gertude" "Montiel") "Gertude Montiel"))
(constraint (= (f "Gertude" "Montiel") "Gertude Montiel"))
(constraint (= (f "Gertude" "Montiel") "Gertude Montiel"))
(constraint (= (f "Stefany" "Reily") "Stefany Reily"))
(constraint (= (f "Stefany" "Reily") "Stefany Reily"))
(constraint (= (f "Stefany" "Reily") "Stefany Reily"))
(constraint (= (f "Rae" "Mcgaughey") "Rae Mcgaughey"))
(constraint (= (f "Rae" "Mcgaughey") "Rae Mcgaughey"))
(constraint (= (f "Rae" "Mcgaughey") "Rae Mcgaughey"))
(constraint (= (f "Cruz" "Latimore") "Cruz Latimore"))
(constraint (= (f "Cruz" "Latimore") "Cruz Latimore"))
(constraint (= (f "Cruz" "Latimore") "Cruz Latimore"))
(constraint (= (f "Maryann" "Casler") "Maryann Casler"))
(constraint (= (f "Maryann" "Casler") "Maryann Casler"))
(constraint (= (f "Maryann" "Casler") "Maryann Casler"))
(constraint (= (f "Annalisa" "Gregori") "Annalisa Gregori"))
(constraint (= (f "Annalisa" "Gregori") "Annalisa Gregori"))
(constraint (= (f "Annalisa" "Gregori") "Annalisa Gregori"))
(constraint (= (f "Jenee" "Pannell") "Jenee Pannell"))
(constraint (= (f "Jenee" "Pannell") "Jenee Pannell"))
(constraint (= (f "Jenee" "Pannell") "Jenee Pannell"))
(constraint (= (f "Launa" "Withers") "Launa Withers"))
(constraint (= (f "Lakenya" "Edison") "Lakenya Edison"))
(constraint (= (f "Brendan" "Hage") "Brendan Hage"))
(constraint (= (f "Bradford" "Lango") "Bradford Lango"))
(constraint (= (f "Rudolf" "Akiyama") "Rudolf Akiyama"))
(constraint (= (f "Lara" "Constable") "Lara Constable"))
(constraint (= (f "Madelaine" "Ghoston") "Madelaine Ghoston"))
(constraint (= (f "Salley" "Hornak") "Salley Hornak"))
(constraint (= (f "Micha" "Junkin") "Micha Junkin"))
(constraint (= (f "Teddy" "Bobo") "Teddy Bobo"))
(constraint (= (f "Coralee" "Scalia") "Coralee Scalia"))
(constraint (= (f "Jeff" "Quashie") "Jeff Quashie"))
(constraint (= (f "Vena" "Babiarz") "Vena Babiarz"))
(constraint (= (f "Karrie" "Lain") "Karrie Lain"))
(constraint (= (f "Tobias" "Dermody") "Tobias Dermody"))
(constraint (= (f "Celsa" "Hopkins") "Celsa Hopkins"))
(constraint (= (f "Kimberley" "Halpern") "Kimberley Halpern"))
(constraint (= (f "Phillip" "Rowden") "Phillip Rowden"))
(constraint (= (f "Elias" "Neil") "Elias Neil"))
(constraint (= (f "Lashanda" "Cortes") "Lashanda Cortes"))
(constraint (= (f "Mackenzie" "Spell") "Mackenzie Spell"))
(constraint (= (f "Kathlyn" "Eccleston") "Kathlyn Eccleston"))
(constraint (= (f "Georgina" "Brescia") "Georgina Brescia"))
(constraint (= (f "Beata" "Miah") "Beata Miah"))
(constraint (= (f "Desiree" "Seamons") "Desiree Seamons"))
(constraint (= (f "Jeanice" "Soderstrom") "Jeanice Soderstrom"))
(constraint (= (f "Mariel" "Jurgens") "Mariel Jurgens"))
(constraint (= (f "Alida" "Bogle") "Alida Bogle"))
(constraint (= (f "Jacqualine" "Olague") "Jacqualine Olague"))
(constraint (= (f "Joaquin" "Clasen") "Joaquin Clasen"))
(constraint (= (f "Samuel" "Richert") "Samuel Richert"))
(constraint (= (f "Malissa" "Marcus") "Malissa Marcus"))
(constraint (= (f "Alaina" "Partida") "Alaina Partida"))
(constraint (= (f "Trinidad" "Mulloy") "Trinidad Mulloy"))
(constraint (= (f "Carlene" "Garrard") "Carlene Garrard"))
(constraint (= (f "Melodi" "Chism") "Melodi Chism"))
(constraint (= (f "Bess" "Chilcott") "Bess Chilcott"))
(constraint (= (f "Chong" "Aylward") "Chong Aylward"))
(constraint (= (f "Jani" "Ramthun") "Jani Ramthun"))
(constraint (= (f "Jacquiline" "Heintz") "Jacquiline Heintz"))
(constraint (= (f "Hayley" "Marquess") "Hayley Marquess"))
(constraint (= (f "Andria" "Spagnoli") "Andria Spagnoli"))
(constraint (= (f "Irwin" "Covelli") "Irwin Covelli"))
(constraint (= (f "Gertude" "Montiel") "Gertude Montiel"))
(constraint (= (f "Stefany" "Reily") "Stefany Reily"))
(constraint (= (f "Rae" "Mcgaughey") "Rae Mcgaughey"))
(constraint (= (f "Cruz" "Latimore") "Cruz Latimore"))
(constraint (= (f "Maryann" "Casler") "Maryann Casler"))
(constraint (= (f "Annalisa" "Gregori") "Annalisa Gregori"))
(constraint (= (f "Jenee" "Pannell") "Jenee Pannell")) 
(check-synth)
