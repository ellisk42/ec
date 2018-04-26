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

(constraint (= (f "+106 769-858-438") "858"))
(constraint (= (f "+83 973-757-831") "757"))
(constraint (= (f "+62 647-787-775") "787"))
(constraint (= (f "+172 027-507-632") "507"))
(constraint (= (f "+72 001-050-856") "050"))
(constraint (= (f "+95 310-537-401") "537"))
(constraint (= (f "+6 775-969-238") "969"))
(constraint (= (f "+174 594-539-946") "539"))
(constraint (= (f "+155 927-275-860") "275"))
(constraint (= (f "+167 405-461-331") "461"))
(constraint (= (f "+10 538-347-401") "347"))
(constraint (= (f "+60 971-986-103") "986"))
(constraint (= (f "+13 258-276-941") "276"))
(constraint (= (f "+2 604-746-137") "746"))
(constraint (= (f "+25 998-898-180") "898"))
(constraint (= (f "+151 862-946-541") "946"))
(constraint (= (f "+118 165-041-038") "041"))
(constraint (= (f "+144 170-592-272") "592"))
(constraint (= (f "+94 462-008-482") "008"))
(constraint (= (f "+82 685-122-086") "122"))
(constraint (= (f "+82 675-366-472") "366"))
(constraint (= (f "+80 066-433-096") "433"))
(constraint (= (f "+163 039-436-166") "436"))
(constraint (= (f "+138 808-083-074") "083"))
(constraint (= (f "+42 643-245-738") "245"))
(constraint (= (f "+169 822-542-726") "542"))
(constraint (= (f "+176 767-782-369") "782"))
(constraint (= (f "+47 414-369-343") "369"))
(constraint (= (f "+138 885-618-512") "618"))
(constraint (= (f "+104 158-671-355") "671"))
(constraint (= (f "+188 280-087-526") "087"))
(constraint (= (f "+50 268-571-336") "571"))
(constraint (= (f "+183 225-960-024") "960"))
(constraint (= (f "+58 191-982-491") "982"))
(constraint (= (f "+9 507-092-535") "092"))
(constraint (= (f "+64 061-601-398") "601"))
(constraint (= (f "+189 831-591-877") "591"))
(constraint (= (f "+129 425-765-844") "765"))
(constraint (= (f "+94 856-734-046") "734"))
(constraint (= (f "+35 082-845-261") "845"))
(constraint (= (f "+185 394-622-272") "622"))
(constraint (= (f "+163 905-707-740") "707"))
(constraint (= (f "+23 448-213-807") "213"))
(constraint (= (f "+42 634-077-089") "077"))
(constraint (= (f "+18 051-287-382") "287"))
(constraint (= (f "+29 773-545-520") "545"))
(constraint (= (f "+43 249-097-743") "097"))
(constraint (= (f "+158 674-736-891") "736"))
(constraint (= (f "+45 124-771-454") "771"))
(constraint (= (f "+180 029-457-654") "457"))
(constraint (= (f "+75 227-250-652") "250"))
(constraint (= (f "+5 528-317-854") "317"))
(constraint (= (f "+81 849-629-290") "629"))
(constraint (= (f "+46 005-119-176") "119"))
(constraint (= (f "+108 150-380-705") "380"))
(constraint (= (f "+40 122-224-247") "224"))
(constraint (= (f "+68 890-680-027") "680"))
(constraint (= (f "+169 060-204-504") "204"))
(constraint (= (f "+95 620-820-945") "820"))
(constraint (= (f "+43 592-938-846") "938"))
(constraint (= (f "+7 023-296-647") "296"))
(constraint (= (f "+20 541-401-396") "401"))
(constraint (= (f "+64 751-365-934") "365"))
(constraint (= (f "+163 546-119-476") "119"))
(constraint (= (f "+198 557-666-779") "666"))
(constraint (= (f "+14 673-759-017") "759"))
(constraint (= (f "+161 086-020-168") "020"))
(constraint (= (f "+65 970-575-488") "575"))
(constraint (= (f "+2 455-126-377") "126"))
(constraint (= (f "+196 728-585-376") "585"))
(constraint (= (f "+33 117-430-125") "430"))
(constraint (= (f "+195 488-831-768") "831"))
(constraint (= (f "+86 468-718-108") "718"))
(constraint (= (f "+194 278-716-950") "716"))
(constraint (= (f "+43 730-685-847") "685"))
(constraint (= (f "+140 794-289-551") "289"))
(constraint (= (f "+21 679-740-834") "740"))
(constraint (= (f "+98 717-997-323") "997"))
(constraint (= (f "+47 401-100-231") "100"))
(constraint (= (f "+143 726-462-368") "462"))
(constraint (= (f "+147 864-005-968") "005"))
(constraint (= (f "+130 590-757-665") "757"))
(constraint (= (f "+197 700-858-976") "858"))
(constraint (= (f "+158 344-541-946") "541"))
(constraint (= (f "+56 242-901-234") "901"))
(constraint (= (f "+132 313-075-754") "075"))
(constraint (= (f "+130 517-953-149") "953"))
(constraint (= (f "+158 684-878-743") "878"))
(constraint (= (f "+52 836-582-035") "582"))
(constraint (= (f "+138 117-484-671") "484"))
(constraint (= (f "+50 012-148-873") "148"))
(constraint (= (f "+105 048-919-483") "919"))
(constraint (= (f "+18 209-851-997") "851"))
(constraint (= (f "+176 938-056-084") "056"))
(constraint (= (f "+141 018-132-973") "132"))
(constraint (= (f "+199 936-162-415") "162"))
(constraint (= (f "+33 547-051-264") "051"))
(constraint (= (f "+161 233-981-513") "981"))
(constraint (= (f "+115 101-728-328") "728"))
(constraint (= (f "+45 095-746-635") "746"))

(check-synth)
