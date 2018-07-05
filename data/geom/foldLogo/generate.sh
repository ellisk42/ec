#!/bin/bash

res=28

for i in {5..6} "ifty"; do
  ../logoDrawString "$res" "fold_${i}_logo_l" 28 "(lambda (logo_forLoop ${i} (lambda (lambda (logo_FWRT (logo_S2L eps) (logo_ADDA (logo_S2A eps) (logo_S2A eps)) \$0))) (\$0)))"
done

for i in {3..4}; do
  ../logoDrawString "$res" "fold2_${i}_logo_l" 28 "(lambda (logo_forLoop ${i} (lambda (lambda (logo_FWRT (logo_S2L (logo_I2S 1)) (logo_DIVA (logo_S2A (logo_I2S 1)) (logo_I2S 2)) \$0))) (\$0)))"
done
