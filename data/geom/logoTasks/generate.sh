#!/bin/bash

rm "./*.png"

resh=512
resl=28

for i in {3..6}; do
  ../logoDrawString $resl "spiral${i}_l" 0 "(lambda ((logo_forLoopM ($i) (lambda (logo_FWRT (logo_MULL (logo_UL) (\$0)) (logo_DIVA (logo_UA) (2)) \$1))) \$0))"
  ../logoDrawString $resl "smooth_spiral${i}_l" 0 "(lambda (logo_forLoopM (logo_IFTY) (lambda (logo_FWRT (logo_MULL logo_epsL \$0) (logo_MULA logo_epsA $i) \$1)) \$0))"

  ../logoDrawString $resh "spiral${i}_h" 0 "(lambda ((logo_forLoopM ($i) (lambda (logo_FWRT (logo_MULL (logo_UL) (\$0)) (logo_DIVA (logo_UA) (2)) \$1))) \$0))"
  ../logoDrawString $resh "smooth_spiral${i}_h" 0 "(lambda (logo_forLoopM (logo_IFTY) (lambda (logo_FWRT (logo_MULL logo_epsL \$0) (logo_MULA logo_epsA $i) \$1)) \$0))"
done

for i in "5" "7"; do
  ../logoDrawString $resl "star_${i}_l" 0 "(lambda (logo_forLoopM ($i) (lambda (logo_FWRT (logo_MULL logo_UL 3) (logo_SUBA (logo_UA) (logo_DIVA (logo_MULA logo_UA 1) $i)) \$1)) \$0))"
  ../logoDrawString $resh "star_${i}_h" 0 "(lambda (logo_forLoopM ($i) (lambda (logo_FWRT (logo_MULL logo_UL 3) (logo_SUBA (logo_UA) (logo_DIVA (logo_MULA logo_UA 1) $i)) \$1)) \$0))" pretty
done

../logoDrawString $resh "iter1_h" 0 "(lambda (logo_forLoopM logo_IFTY (lambda (logo_FWRT logo_epsL (logo_DIVA logo_epsA 2) \$1)) \$0))"
../logoDrawString $resh "iter2_h" 0 "(lambda (logo_forLoopM 2 (lambda (logo_forLoopM logo_IFTY (lambda (logo_FWRT logo_epsL (logo_DIVA logo_epsA 2) \$2)) (logo_FWRT (logo_MULL logo_UL 0) (logo_DIVA logo_UA 2) \$1))) \$0))"
../logoDrawString $resl "iter1_l" 0 "(lambda (logo_forLoopM logo_IFTY (lambda (logo_FWRT logo_epsL (logo_DIVA logo_epsA 2) \$1)) \$0))"
../logoDrawString $resl "iter2_l" 0 "(lambda (logo_forLoopM 2 (lambda (logo_forLoopM logo_IFTY (lambda (logo_FWRT logo_epsL (logo_DIVA logo_epsA 2) \$2)) (logo_FWRT (logo_MULL logo_UL 0) (logo_DIVA logo_UA 2) \$1))) \$0))"

for i in "5" "7"; do
  ../logoDrawString $resh "flower_${i}_h" 0 "(lambda (logo_forLoopM $i (lambda (logo_forLoopM 2 (lambda (logo_forLoopM logo_IFTY (lambda (logo_FWRT logo_epsL (logo_DIVA logo_epsA 2) \$3)) (logo_FWRT (logo_MULL logo_UL 0) (logo_DIVA logo_UA 2) \$2))) (logo_FWRT (logo_MULL logo_UL 0) (logo_DIVA (logo_MULA logo_UA 2) $i) \$1))) \$0))"
  ../logoDrawString $resl "flower_${i}_l" 0 "(lambda (logo_forLoopM $i (lambda (logo_forLoopM 2 (lambda (logo_forLoopM logo_IFTY (lambda (logo_FWRT logo_epsL (logo_DIVA logo_epsA 2) \$3)) (logo_FWRT (logo_MULL logo_UL 0) (logo_DIVA logo_UA 2) \$2))) (logo_FWRT (logo_MULL logo_UL 0) (logo_DIVA (logo_MULA logo_UA 2) $i) \$1))) \$0))"
done


