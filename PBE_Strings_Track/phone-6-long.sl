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


(constraint (= (f "+106 769-858-438") "769"))
(constraint (= (f "+83 973-757-831") "973"))
(constraint (= (f "+62 647-787-775") "647"))
(constraint (= (f "+172 027-507-632") "027"))
(constraint (= (f "+72 001-050-856") "001"))
(constraint (= (f "+95 310-537-401") "310"))
(constraint (= (f "+6 775-969-238") "775"))
(constraint (= (f "+174 594-539-946") "594"))
(constraint (= (f "+155 927-275-860") "927"))
(constraint (= (f "+167 405-461-331") "405"))
(constraint (= (f "+10 538-347-401") "538"))
(constraint (= (f "+60 971-986-103") "971"))
(constraint (= (f "+13 258-276-941") "258"))
(constraint (= (f "+2 604-746-137") "604"))
(constraint (= (f "+25 998-898-180") "998"))
(constraint (= (f "+151 862-946-541") "862"))
(constraint (= (f "+118 165-041-038") "165"))
(constraint (= (f "+144 170-592-272") "170"))
(constraint (= (f "+94 462-008-482") "462"))
(constraint (= (f "+82 685-122-086") "685"))
(constraint (= (f "+82 675-366-472") "675"))
(constraint (= (f "+80 066-433-096") "066"))
(constraint (= (f "+163 039-436-166") "039"))
(constraint (= (f "+138 808-083-074") "808"))
(constraint (= (f "+42 643-245-738") "643"))
(constraint (= (f "+169 822-542-726") "822"))
(constraint (= (f "+176 767-782-369") "767"))
(constraint (= (f "+47 414-369-343") "414"))
(constraint (= (f "+138 885-618-512") "885"))
(constraint (= (f "+104 158-671-355") "158"))
(constraint (= (f "+188 280-087-526") "280"))
(constraint (= (f "+50 268-571-336") "268"))
(constraint (= (f "+183 225-960-024") "225"))
(constraint (= (f "+58 191-982-491") "191"))
(constraint (= (f "+9 507-092-535") "507"))
(constraint (= (f "+64 061-601-398") "061"))
(constraint (= (f "+189 831-591-877") "831"))
(constraint (= (f "+129 425-765-844") "425"))
(constraint (= (f "+94 856-734-046") "856"))
(constraint (= (f "+35 082-845-261") "082"))
(constraint (= (f "+185 394-622-272") "394"))
(constraint (= (f "+163 905-707-740") "905"))
(constraint (= (f "+23 448-213-807") "448"))
(constraint (= (f "+42 634-077-089") "634"))
(constraint (= (f "+18 051-287-382") "051"))
(constraint (= (f "+29 773-545-520") "773"))
(constraint (= (f "+43 249-097-743") "249"))
(constraint (= (f "+158 674-736-891") "674"))
(constraint (= (f "+45 124-771-454") "124"))
(constraint (= (f "+180 029-457-654") "029"))
(constraint (= (f "+75 227-250-652") "227"))
(constraint (= (f "+5 528-317-854") "528"))
(constraint (= (f "+81 849-629-290") "849"))
(constraint (= (f "+46 005-119-176") "005"))
(constraint (= (f "+108 150-380-705") "150"))
(constraint (= (f "+40 122-224-247") "122"))
(constraint (= (f "+68 890-680-027") "890"))
(constraint (= (f "+169 060-204-504") "060"))
(constraint (= (f "+95 620-820-945") "620"))
(constraint (= (f "+43 592-938-846") "592"))
(constraint (= (f "+7 023-296-647") "023"))
(constraint (= (f "+20 541-401-396") "541"))
(constraint (= (f "+64 751-365-934") "751"))
(constraint (= (f "+163 546-119-476") "546"))
(constraint (= (f "+198 557-666-779") "557"))
(constraint (= (f "+14 673-759-017") "673"))
(constraint (= (f "+161 086-020-168") "086"))
(constraint (= (f "+65 970-575-488") "970"))
(constraint (= (f "+2 455-126-377") "455"))
(constraint (= (f "+196 728-585-376") "728"))
(constraint (= (f "+33 117-430-125") "117"))
(constraint (= (f "+195 488-831-768") "488"))
(constraint (= (f "+86 468-718-108") "468"))
(constraint (= (f "+194 278-716-950") "278"))
(constraint (= (f "+43 730-685-847") "730"))
(constraint (= (f "+140 794-289-551") "794"))
(constraint (= (f "+21 679-740-834") "679"))
(constraint (= (f "+98 717-997-323") "717"))
(constraint (= (f "+47 401-100-231") "401"))
(constraint (= (f "+143 726-462-368") "726"))
(constraint (= (f "+147 864-005-968") "864"))
(constraint (= (f "+130 590-757-665") "590"))
(constraint (= (f "+197 700-858-976") "700"))
(constraint (= (f "+158 344-541-946") "344"))
(constraint (= (f "+56 242-901-234") "242"))
(constraint (= (f "+132 313-075-754") "313"))
(constraint (= (f "+130 517-953-149") "517"))
(constraint (= (f "+158 684-878-743") "684"))
(constraint (= (f "+52 836-582-035") "836"))
(constraint (= (f "+138 117-484-671") "117"))
(constraint (= (f "+50 012-148-873") "012"))
(constraint (= (f "+105 048-919-483") "048"))
(constraint (= (f "+18 209-851-997") "209"))
(constraint (= (f "+176 938-056-084") "938"))
(constraint (= (f "+141 018-132-973") "018"))
(constraint (= (f "+199 936-162-415") "936"))
(constraint (= (f "+33 547-051-264") "547"))
(constraint (= (f "+161 233-981-513") "233"))
(constraint (= (f "+115 101-728-328") "101"))
(constraint (= (f "+45 095-746-635") "095"))

(check-synth)
