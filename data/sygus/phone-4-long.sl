(set-logic SLIA)
 
(synth-fun f ((name String)) String
    ((Start String (ntString))
     (ntString String (name " " "(" ")" "-" "."
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

(constraint (= (f "938-242-504") "938.242.504"))
(constraint (= (f "308-916-545") "308.916.545"))
(constraint (= (f "623-599-749") "623.599.749"))
(constraint (= (f "981-424-843") "981.424.843"))
(constraint (= (f "118-980-214") "118.980.214"))
(constraint (= (f "244-655-094") "244.655.094"))
(constraint (= (f "830-941-991") "830.941.991"))
(constraint (= (f "911-186-562") "911.186.562"))
(constraint (= (f "002-500-200") "002.500.200"))
(constraint (= (f "113-860-034") "113.860.034"))
(constraint (= (f "457-622-959") "457.622.959"))
(constraint (= (f "986-722-311") "986.722.311"))
(constraint (= (f "110-170-771") "110.170.771"))
(constraint (= (f "469-610-118") "469.610.118"))
(constraint (= (f "817-925-247") "817.925.247"))
(constraint (= (f "256-899-439") "256.899.439"))
(constraint (= (f "886-911-726") "886.911.726"))
(constraint (= (f "562-950-358") "562.950.358"))
(constraint (= (f "693-049-588") "693.049.588"))
(constraint (= (f "840-503-234") "840.503.234"))
(constraint (= (f "698-815-340") "698.815.340"))
(constraint (= (f "498-808-434") "498.808.434"))
(constraint (= (f "329-545-000") "329.545.000"))
(constraint (= (f "380-281-597") "380.281.597"))
(constraint (= (f "332-395-493") "332.395.493"))
(constraint (= (f "251-903-028") "251.903.028"))
(constraint (= (f "176-090-894") "176.090.894"))
(constraint (= (f "336-611-100") "336.611.100"))
(constraint (= (f "416-390-647") "416.390.647"))
(constraint (= (f "019-430-596") "019.430.596"))
(constraint (= (f "960-659-771") "960.659.771"))
(constraint (= (f "475-505-007") "475.505.007"))
(constraint (= (f "424-069-886") "424.069.886"))
(constraint (= (f "941-102-117") "941.102.117"))
(constraint (= (f "331-728-008") "331.728.008"))
(constraint (= (f "487-726-198") "487.726.198"))
(constraint (= (f "612-419-942") "612.419.942"))
(constraint (= (f "594-741-346") "594.741.346"))
(constraint (= (f "320-984-742") "320.984.742"))
(constraint (= (f "060-919-361") "060.919.361"))
(constraint (= (f "275-536-998") "275.536.998"))
(constraint (= (f "548-835-065") "548.835.065"))
(constraint (= (f "197-485-507") "197.485.507"))
(constraint (= (f "455-776-949") "455.776.949"))
(constraint (= (f "085-421-340") "085.421.340"))
(constraint (= (f "785-713-099") "785.713.099"))
(constraint (= (f "426-712-861") "426.712.861"))
(constraint (= (f "386-994-906") "386.994.906"))
(constraint (= (f "918-304-840") "918.304.840"))
(constraint (= (f "247-153-598") "247.153.598"))
(constraint (= (f "075-497-069") "075.497.069"))
(constraint (= (f "140-726-583") "140.726.583"))
(constraint (= (f "049-413-248") "049.413.248"))
(constraint (= (f "977-386-462") "977.386.462"))
(constraint (= (f "058-272-455") "058.272.455"))
(constraint (= (f "428-629-927") "428.629.927"))
(constraint (= (f "449-122-191") "449.122.191"))
(constraint (= (f "568-759-670") "568.759.670"))
(constraint (= (f "312-846-053") "312.846.053"))
(constraint (= (f "943-037-297") "943.037.297"))
(constraint (= (f "014-270-177") "014.270.177"))
(constraint (= (f "658-877-878") "658.877.878"))
(constraint (= (f "888-594-038") "888.594.038"))
(constraint (= (f "232-253-254") "232.253.254"))
(constraint (= (f "308-722-292") "308.722.292"))
(constraint (= (f "342-145-742") "342.145.742"))
(constraint (= (f "568-181-515") "568.181.515"))
(constraint (= (f "300-140-756") "300.140.756"))
(constraint (= (f "099-684-216") "099.684.216"))
(constraint (= (f "575-296-621") "575.296.621"))
(constraint (= (f "994-443-794") "994.443.794"))
(constraint (= (f "400-334-692") "400.334.692"))
(constraint (= (f "684-711-883") "684.711.883"))
(constraint (= (f "539-636-358") "539.636.358"))
(constraint (= (f "009-878-919") "009.878.919"))
(constraint (= (f "919-545-701") "919.545.701"))
(constraint (= (f "546-399-239") "546.399.239"))
(constraint (= (f "993-608-757") "993.608.757"))
(constraint (= (f "107-652-845") "107.652.845"))
(constraint (= (f "206-805-793") "206.805.793"))
(constraint (= (f "198-857-684") "198.857.684"))
(constraint (= (f "912-827-430") "912.827.430"))
(constraint (= (f "560-951-766") "560.951.766"))
(constraint (= (f "142-178-290") "142.178.290"))
(constraint (= (f "732-196-946") "732.196.946"))
(constraint (= (f "963-875-745") "963.875.745"))
(constraint (= (f "881-865-867") "881.865.867"))
(constraint (= (f "234-686-715") "234.686.715"))
(constraint (= (f "720-330-583") "720.330.583"))
(constraint (= (f "593-065-126") "593.065.126"))
(constraint (= (f "671-778-064") "671.778.064"))
(constraint (= (f "252-029-036") "252.029.036"))
(constraint (= (f "700-322-036") "700.322.036"))
(constraint (= (f "882-587-473") "882.587.473"))
(constraint (= (f "964-134-953") "964.134.953"))
(constraint (= (f "038-300-876") "038.300.876"))
(constraint (= (f "158-894-947") "158.894.947"))
(constraint (= (f "757-454-374") "757.454.374"))
(constraint (= (f "872-513-190") "872.513.190"))
(constraint (= (f "566-086-726") "566.086.726"))

(check-synth)
