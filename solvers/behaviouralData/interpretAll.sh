#!/bin/bash

for file in *.LoG
do
  echo "Drawing file $file"
  ./geomDrawFile $file
done
