#!/bin/bash

FILE=$1;
export ID=$(echo ${FILE} | cut -d '_' -f 1);
cat ${FILE} | jq -c '. | {id: $ENV.ID, program: .concept, data: .examples}' | underscore print -o ${FILE}
