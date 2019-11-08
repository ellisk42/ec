(set-logic SLIA)
 
(synth-fun f ((name String)) String
    ((Start String (ntString))
     (ntString String (name " " "+" "-" "."
                       (str.++ ntString ntString)
                       (str.replace ntString ntString ntString)
                       (str.at ntString ntInt)
                       (int.to.str ntInt)
                       (str.substr ntString ntInt ntInt)))
      (ntInt Int (0 1 2 3 4 5
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

(constraint (= (f "+106 769-858-438") "438"))
(constraint (= (f "+83 973-757-831") "831"))
(constraint (= (f "+62 647-787-775") "775"))
(constraint (= (f "+172 027-507-632") "632"))
(constraint (= (f "+72 001-050-856") "856"))
(constraint (= (f "+95 310-537-401") "401"))
(constraint (= (f "+6 775-969-238") "238"))
(constraint (= (f "+174 594-539-946") "946"))
(constraint (= (f "+155 927-275-860") "860"))
(constraint (= (f "+167 405-461-331") "331"))
(constraint (= (f "+10 538-347-401") "401"))
(constraint (= (f "+60 971-986-103") "103"))
(constraint (= (f "+13 258-276-941") "941"))
(constraint (= (f "+2 604-746-137") "137"))
(constraint (= (f "+25 998-898-180") "180"))
(constraint (= (f "+151 862-946-541") "541"))
(constraint (= (f "+118 165-041-038") "038"))
(constraint (= (f "+144 170-592-272") "272"))
(constraint (= (f "+94 462-008-482") "482"))
(constraint (= (f "+82 685-122-086") "086"))
(constraint (= (f "+82 675-366-472") "472"))
(constraint (= (f "+80 066-433-096") "096"))
(constraint (= (f "+163 039-436-166") "166"))
(constraint (= (f "+138 808-083-074") "074"))
(constraint (= (f "+42 643-245-738") "738"))
(constraint (= (f "+169 822-542-726") "726"))
(constraint (= (f "+176 767-782-369") "369"))
(constraint (= (f "+47 414-369-343") "343"))
(constraint (= (f "+138 885-618-512") "512"))
(constraint (= (f "+104 158-671-355") "355"))
(constraint (= (f "+188 280-087-526") "526"))
(constraint (= (f "+50 268-571-336") "336"))
(constraint (= (f "+183 225-960-024") "024"))
(constraint (= (f "+58 191-982-491") "491"))
(constraint (= (f "+9 507-092-535") "535"))
(constraint (= (f "+64 061-601-398") "398"))
(constraint (= (f "+189 831-591-877") "877"))
(constraint (= (f "+129 425-765-844") "844"))
(constraint (= (f "+94 856-734-046") "046"))
(constraint (= (f "+35 082-845-261") "261"))
(constraint (= (f "+185 394-622-272") "272"))
(constraint (= (f "+163 905-707-740") "740"))
(constraint (= (f "+23 448-213-807") "807"))
(constraint (= (f "+42 634-077-089") "089"))
(constraint (= (f "+18 051-287-382") "382"))
(constraint (= (f "+29 773-545-520") "520"))
(constraint (= (f "+43 249-097-743") "743"))
(constraint (= (f "+158 674-736-891") "891"))
(constraint (= (f "+45 124-771-454") "454"))
(constraint (= (f "+180 029-457-654") "654"))
(constraint (= (f "+75 227-250-652") "652"))
(constraint (= (f "+5 528-317-854") "854"))
(constraint (= (f "+81 849-629-290") "290"))
(constraint (= (f "+46 005-119-176") "176"))
(constraint (= (f "+108 150-380-705") "705"))
(constraint (= (f "+40 122-224-247") "247"))
(constraint (= (f "+68 890-680-027") "027"))
(constraint (= (f "+169 060-204-504") "504"))
(constraint (= (f "+95 620-820-945") "945"))
(constraint (= (f "+43 592-938-846") "846"))
(constraint (= (f "+7 023-296-647") "647"))
(constraint (= (f "+20 541-401-396") "396"))
(constraint (= (f "+64 751-365-934") "934"))
(constraint (= (f "+163 546-119-476") "476"))
(constraint (= (f "+198 557-666-779") "779"))
(constraint (= (f "+14 673-759-017") "017"))
(constraint (= (f "+161 086-020-168") "168"))
(constraint (= (f "+65 970-575-488") "488"))
(constraint (= (f "+2 455-126-377") "377"))
(constraint (= (f "+196 728-585-376") "376"))
(constraint (= (f "+33 117-430-125") "125"))
(constraint (= (f "+195 488-831-768") "768"))
(constraint (= (f "+86 468-718-108") "108"))
(constraint (= (f "+194 278-716-950") "950"))
(constraint (= (f "+43 730-685-847") "847"))
(constraint (= (f "+140 794-289-551") "551"))
(constraint (= (f "+21 679-740-834") "834"))
(constraint (= (f "+98 717-997-323") "323"))
(constraint (= (f "+47 401-100-231") "231"))
(constraint (= (f "+143 726-462-368") "368"))
(constraint (= (f "+147 864-005-968") "968"))
(constraint (= (f "+130 590-757-665") "665"))
(constraint (= (f "+197 700-858-976") "976"))
(constraint (= (f "+158 344-541-946") "946"))
(constraint (= (f "+56 242-901-234") "234"))
(constraint (= (f "+132 313-075-754") "754"))
(constraint (= (f "+130 517-953-149") "149"))
(constraint (= (f "+158 684-878-743") "743"))
(constraint (= (f "+52 836-582-035") "035"))
(constraint (= (f "+138 117-484-671") "671"))
(constraint (= (f "+50 012-148-873") "873"))
(constraint (= (f "+105 048-919-483") "483"))
(constraint (= (f "+18 209-851-997") "997"))
(constraint (= (f "+176 938-056-084") "084"))
(constraint (= (f "+141 018-132-973") "973"))
(constraint (= (f "+199 936-162-415") "415"))
(constraint (= (f "+33 547-051-264") "264"))
(constraint (= (f "+161 233-981-513") "513"))
(constraint (= (f "+115 101-728-328") "328"))
(constraint (= (f "+45 095-746-635") "635"))

(check-synth)
