#!/bin/bash

res=28

for i in {4..6} "20"; do
  ../logoDrawString "$res" "fold_${i}_logo_l" 28 "(lambda (fold (range ${i}) \$0 (lambda (lambda (logo_FWRT (logo_S2L eps) (logo_S2A eps) \$0)))))"
  ../logoDrawString "$res" "fold2_${i}_logo_l" 28 "(lambda (fold (range ${i}) \$0 (lambda (lambda (line (logo_FWRT (logo_S2L (logo_I2S \$1)) (logo_S2A (logo_I2S 1)) \$0))))))"
done

