#!/bin/bash

for i in {2..6}; do
  for e in "3" "4" "5"; do
    ../logoDrawString 28 "c${e}n${i}_l.png" 28 "(fold (range $i) (logo_NOP) (lambda (lambda (logo_SEQ (logo_SEQ (logo_FW (logo_S2L (logo_I2S 1))) (logo_RT (logo_DIVA (logo_S2A (logo_I2S 1)) (logo_I2S $e)))) (\$0)))))"
  done
done
