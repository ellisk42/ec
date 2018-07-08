#!/bin/bash

resh=512
resl=28

for i in {3..6}; do
  ../logoDrawString $resl "spiral${i}_l" 0 "(lambda ((logo_forLoopM ($i) (lambda (logo_FWRT (logo_MULL (logo_UL) (\$0)) (logo_DIVA (logo_UA) (2)) \$1))) \$0))"
  ../logoDrawString $resl "smooth_spiral${i}_l" 0 "(lambda (logo_forLoopM (logo_IFTY) (lambda (logo_FWRT (logo_MULL (logo_DIVL logo_epsL 2) \$0) (logo_MULA logo_epsA $i) \$1)) \$0))"

  ../logoDrawString $resh "spiral${i}_h" 0 "(lambda ((logo_forLoopM ($i) (lambda (logo_FWRT (logo_MULL (logo_UL) (\$0)) (logo_DIVA (logo_UA) (2)) \$1))) \$0))"
  ../logoDrawString $resh "smooth_spiral${i}_h" 0 "(lambda (logo_forLoopM (logo_IFTY) (lambda (logo_FWRT (logo_MULL (logo_DIVL logo_epsL 2) \$0) (logo_MULA logo_epsA $i) \$1)) \$0))"
done

