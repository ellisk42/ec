#!/bin/bash

for i in 5 7 8 9; do
  ../logoDrawString 28 "n${i}_l" 28 "(fold (range $i) (logo_NOP) (lambda (lambda (logo_SEQ (logo_SEQ (logo_FW (logo_S2L (logo_I2S \$1))) (logo_RT (logo_DIVA (logo_S2A (logo_I2S 1)) (logo_I2S 2)))) (\$0)))))"
done

../logoDrawString 28 "segment_l" 28 "(logo_FW (logo_S2L (logo_I2S 1)))"
