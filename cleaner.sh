#!/bin/bash

BAR="████████████████████████████████████████████████████████████"

ITER=0

for d in ./1530*/; do

  echo "Taking care of folder $d:"

  rm -Rf "$ITER"
  mkdir -p "$ITER"
  cp "$d"*.png "$ITER"
  fdupes -dqI "$ITER" > /dev/null

  N1="$(find "$d" -type f -iname "*.png" | wc -l)"
  N2="$(find "$ITER" -type f -iname "*.png" | wc -l)"

  echo "1 - (${N2} / ${N1})" | bc -ql | cut -c 2-3 | sed -e 's/\(.*\)/  \1% dreams removed due to duplication/g'
  echo "  $N2 images remaining"

  #echo "  Generating noisy versions:"

  #i=0
  #for f in $ITER/*.png; do
    #p="$((60*i/N2))"
    #p2="$((100*i/N2))"
    #((i+=1))
    #echo -ne "\\r    Progress: $p2% ${BAR:0:$p}"
    #NAME="$(basename "${f%.*}")"
    #cp "$d/$NAME.LoG" "$d/$NAME.dream" "$ITER"
    #mv "$ITER/$NAME.png" "$ITER/${NAME}_origin.png"
    #pids=""
    #./geomDrawLambdaString "$ITER/${NAME}_r1.png" "noise" "$(cat "$ITER/$NAME.LoG")" &
    #pids="$pids $!"
    #./geomDrawLambdaString "$ITER/${NAME}_r2.png" "noise" "$(cat "$ITER/$NAME.LoG")" &
    #pids="$pids $!"
    #./geomDrawLambdaString "$ITER/${NAME}_r3.png" "noise" "$(cat "$ITER/$NAME.LoG")" &
    #pids="$pids $!"
    #./geomDrawLambdaString "$ITER/${NAME}_r4.png" "noise" "$(cat "$ITER/$NAME.LoG")" &
    #pids="$pids $!"
    #./geomDrawLambdaString "$ITER/${NAME}_r5.png" "noise" "$(cat "$ITER/$NAME.LoG")" &
    #pids="$pids $!"
    #wait $pids
  #done

  #echo -e "\\r    Progress: 100% ${BAR:0:59}"

  echo "  Generating montages."
  pids=""
  montage "$ITER"/*.png "$ITER"_montage.png &
  pids="$pids $!"
  #montage "$ITER"/*_r1.png "$ITER"_montage_r1.png &
  #pids="$pids $!"
  #montage "$ITER"/*_r2.png "$ITER"_montage_r2.png &
  #pids="$pids $!"
  #montage "$ITER"/*_r3.png "$ITER"_montage_r3.png &
  #pids="$pids $!"
  #montage "$ITER"/*_r4.png "$ITER"_montage_r4.png &
  #pids="$pids $!"
  #montage "$ITER"/*_r5.png "$ITER"_montage_r5.png &
  #pids="$pids $!"
  wait $pids


  ((ITER+=1))
done

