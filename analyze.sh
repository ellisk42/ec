
for model in "base" "bayes" "vs" "bayes_vs"; do
  echo "Model: $model"
  TIME0=$(grep "seconds" "artifact_out/$model-0.stderr" | tail -n1 | awk '{print $1}')
  TIME1=$(grep "seconds" "artifact_out/$model-1.stderr" | tail -n1 | awk '{print $1}')
  TIME2=$(grep "seconds" "artifact_out/$model-2.stderr" | tail -n1 | awk '{print $1}')
  TIME3=$(grep "seconds" "artifact_out/$model-3.stderr" | tail -n1 | awk '{print $1}')
  TIME=$(echo $TIME0+$TIME1+$TIME2+$TIME3 | bc)
  echo "Runtime: $TIME"

  echo "FOLD"
  grep "^(t0 -> t1 -> t1) -> t1 -> list(t0) -> t1" artifact_out/$model-0.stderr | tail -n1
  grep "^t0 -> (t1 -> t0 -> t0) -> list(t1) -> t0" artifact_out/$model-0.stderr | tail -n1

  echo "UNFOLD"
  grep "^(t0 -> t0) -> (t0 -> t1) -> (t0 -> bool) -> t0 -> list(t1)" artifact_out/$model-0.stderr | tail -n1

  echo "MAP"
  grep "^(t0 -> t1) -> list(t0) -> list(t1)" artifact_out/$model-0.stderr | tail -n1

  echo "FILTER"
  grep "^(t0 -> bool) -> list(t0) -> list(t0)" artifact_out/$model-1.stderr | tail -n1

  echo "ZIP"
  grep "^(t0 -> t1 -> t2) -> list(t0) -> list(t1) -> list(t2)" artifact_out/$model-3.stderr | tail -n1
  
  echo ""
done
